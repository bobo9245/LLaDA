"""Normalize local copies of HarmBench / XSTest / Alpaca-style data into a
single JSONL manifest schema::

    {
      "id":               "<stable-unique-id>",
      "source":           "harmbench" | "xstest" | "alpaca" | ...,
      "prompt":           "<raw user prompt>",
      "messages":         [{"role": "user", "content": "..."}],
      "intended_group":   "T_plus" | "T_minus" | "HN_adjacent" | "C" | "unknown",
      "subconcept":       "<topic tag>" | null,
      "split_hint":       "train" | "val" | "test" | "unknown",
      "length_bucket":    "short" | "medium" | "long"
    }

The normalization rules below are deliberately simple — they only try to
recognize the shape of the standard releases we care about. Users with custom
splits can point ``--schema-overrides`` at a small Python file that returns a
callable and own the mapping.

This script does not download anything. Data must already exist locally.
Pass ``--source`` to control how rows are parsed from ``--input-path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from trajectory_subspace.io_utils import write_jsonl


SHORT_LEN = 80
MEDIUM_LEN = 256


def _length_bucket(text: str) -> str:
    n = len(text)
    if n <= SHORT_LEN:
        return "short"
    if n <= MEDIUM_LEN:
        return "medium"
    return "long"


def _stable_id(source: str, prompt: str, suffix: Optional[str] = None) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    if suffix:
        h.update(b"\x00")
        h.update(suffix.encode("utf-8"))
    return f"{source}-{h.hexdigest()[:16]}"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    import csv

    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _iter_input(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
    elif path.suffix.lower() == ".csv":
        yield from _iter_csv(path)
    else:
        raise ValueError(
            f"Unsupported input format: {path.suffix!r}. Use .jsonl or .csv."
        )


# ---------------------------------------------------------------------------
# Source-specific mappers
# ---------------------------------------------------------------------------


def _map_harmbench(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = row.get("Behavior") or row.get("behavior") or row.get("prompt")
    if not prompt:
        return None
    category = (
        row.get("SemanticCategory")
        or row.get("semantic_category")
        or row.get("category")
    )
    split = row.get("split") or row.get("Split") or "unknown"
    rid = row.get("BehaviorID") or row.get("id") or _stable_id("harmbench", prompt)
    return {
        "id": str(rid),
        "source": "harmbench",
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "intended_group": "T_plus",
        "subconcept": category,
        "split_hint": split,
        "length_bucket": _length_bucket(prompt),
    }


def _map_xstest(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = row.get("prompt") or row.get("Prompt")
    if not prompt:
        return None
    type_ = (row.get("type") or row.get("Type") or "").lower()
    # XSTest has "safe" (should comply) and "unsafe" (should refuse) subsets.
    # The safe-but-triggers-refusal set maps cleanly to HN_adjacent.
    if "contrast" in type_ or "safe" in type_:
        group = "HN_adjacent"
    else:
        group = "T_plus"
    category = row.get("category") or row.get("Category")
    rid = row.get("id") or _stable_id("xstest", prompt)
    return {
        "id": str(rid),
        "source": "xstest",
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "intended_group": group,
        "subconcept": category,
        "split_hint": "unknown",
        "length_bucket": _length_bucket(prompt),
    }


def _map_alpaca(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    instruction = row.get("instruction") or row.get("prompt")
    if not instruction:
        return None
    if row.get("input"):
        prompt = f"{instruction}\n\n{row['input']}"
    else:
        prompt = instruction
    rid = row.get("id") or _stable_id("alpaca", prompt)
    return {
        "id": str(rid),
        "source": "alpaca",
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "intended_group": "C",
        "subconcept": row.get("category"),
        "split_hint": row.get("split", "unknown"),
        "length_bucket": _length_bucket(prompt),
    }


def _map_generic(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = row.get("prompt") or row.get("instruction") or row.get("text")
    if not prompt:
        return None
    rid = row.get("id") or _stable_id("generic", prompt)
    return {
        "id": str(rid),
        "source": "generic",
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "intended_group": row.get("intended_group", "unknown"),
        "subconcept": row.get("subconcept") or row.get("category"),
        "split_hint": row.get("split", "unknown"),
        "length_bucket": _length_bucket(prompt),
    }


MAPPERS: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
    "harmbench": _map_harmbench,
    "xstest": _map_xstest,
    "alpaca": _map_alpaca,
    "generic": _map_generic,
}


def normalize(
    rows: Iterable[Dict[str, Any]], mapper: Callable
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        mapped = mapper(row)
        if mapped is not None:
            yield mapped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-path", required=True, type=Path)
    p.add_argument("--source", required=True, choices=sorted(MAPPERS))
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on rows emitted; useful for pilot runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapper = MAPPERS[args.source]
    src_rows = list(_iter_input(args.input_path))
    normalized = list(normalize(src_rows, mapper))
    if args.limit is not None:
        normalized = normalized[: args.limit]
    write_jsonl(args.output_path, normalized)
    print(
        f"Wrote {len(normalized)} rows (from {len(src_rows)} input rows) "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()
