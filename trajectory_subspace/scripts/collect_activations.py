"""Run LLaDA sampling over a labeled manifest and persist activations.

For every ``(prompt, seed)`` pair we:

1. Instantiate a :class:`trajectory_subspace.observers.CollectObserver` with
   the configured target ratios, layers, and pooling strategies.
2. Run :func:`reverse_diffusion_sample` once.
3. Flush the observer's captured ``(target, layer, pool)`` cells to sharded
   safetensors under ``<out-root>/activations/pooled/<cell>/``.
4. Optionally dump a token-level hidden-state subset and (if requested) a
   very small gradient subset.

Resuming: if a matching ``(prompt_id, seed, target, layer, pool)`` row already
exists in ``index.jsonl``, that sample is skipped. This makes it safe to rerun
after a crash.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from trajectory_subspace.io_utils import (
    ShardWriter,
    read_jsonl,
    yaml_load,
)
from trajectory_subspace.observers import CollectObserver
from trajectory_subspace.sampling import SamplingConfig, reverse_diffusion_sample


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(config: Dict[str, Any]):
    from transformers import AutoModel, AutoTokenizer

    m_cfg = config["model"]
    dtype = getattr(torch, m_cfg.get("dtype", "bfloat16"))
    model = (
        AutoModel.from_pretrained(
            m_cfg["path"],
            trust_remote_code=m_cfg.get("trust_remote_code", True),
            torch_dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["path"], trust_remote_code=m_cfg.get("trust_remote_code", True)
    )
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    return model, tokenizer


def _writers_for_cells(
    out_root: Path,
    group: str,
    target_ratios: List[float],
    layers: List[int],
    poolings: List[str],
    shard_max_rows: int,
    subdir: str = "pooled",
) -> Dict[Tuple[int, int, str], ShardWriter]:
    writers: Dict[Tuple[int, int, str], ShardWriter] = {}
    for ti, t in enumerate(target_ratios):
        for l in layers:
            for p in poolings:
                cell_dir = (
                    out_root
                    / "activations"
                    / subdir
                    / group
                    / f"t={t:.2f}_l={l:02d}_p={p}"
                )
                writers[(ti, l, p)] = ShardWriter(cell_dir, shard_max_rows=shard_max_rows)
    return writers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--labeled-manifest", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="pilot",
        help="Select sampling budget from the config file.",
    )
    p.add_argument(
        "--group",
        default="all",
        help="Output group tag. Used to namespace activation directories.",
    )
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--keep-token-level", action="store_true")
    p.add_argument("--keep-response-logits", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml_load(args.config)
    profile = cfg[args.profile]

    manifest = list(read_jsonl(args.labeled_manifest))
    max_prompts = args.limit_prompts or profile.get("max_prompts")
    if max_prompts is not None:
        manifest = manifest[: max_prompts]

    seeds = cfg["seeds"][: profile.get("max_seeds", len(cfg["seeds"]))]

    target_ratios = list(cfg["capture"]["ratios"])
    layers = list(cfg["capture"]["layers"])
    poolings = list(cfg["capture"]["pooling"])

    sampling = cfg["sampling"]
    shard_max_rows = cfg["io"].get("shard_max_rows", 512)

    args.out_root.mkdir(parents=True, exist_ok=True)
    writers = _writers_for_cells(
        args.out_root,
        args.group,
        target_ratios,
        layers,
        poolings,
        shard_max_rows,
        subdir="pooled",
    )
    token_writers: Dict[Tuple[int, int, str], ShardWriter] = {}
    if args.keep_token_level:
        token_writers = _writers_for_cells(
            args.out_root,
            args.group,
            target_ratios,
            layers,
            poolings,
            shard_max_rows,
            subdir="token_level",
        )

    model, tokenizer = _load_model(cfg)

    def _already_captured(prompt_id: str, seed: int) -> bool:
        # Cheap check using the first cell only — collection is all-or-nothing
        # per (prompt, seed) pair.
        probe = writers[(0, layers[0], poolings[0])]
        return probe.already_has(
            lambda r: r.get("prompt_id") == prompt_id and r.get("seed") == seed
        )

    mask_id = int(sampling.get("mask_id", 126336))
    for row in manifest:
        prompt_id = row["id"]
        messages = row["messages"]
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        for seed in seeds:
            if _already_captured(prompt_id, seed):
                continue
            _set_seed(seed)
            observer = CollectObserver(
                target_ratios=target_ratios,
                layers=layers,
                poolings=poolings,
                keep_token_level=args.keep_token_level,
                keep_response_logits=args.keep_response_logits,
            )
            s_cfg = SamplingConfig(
                steps=sampling["steps"],
                gen_length=sampling["gen_length"],
                block_length=sampling["block_length"],
                temperature=sampling.get("temperature", 0.0),
                cfg_scale=sampling.get("cfg_scale", 0.0),
                remasking=sampling.get("remasking", "low_confidence"),
                mask_id=mask_id,
                capture_layers=layers,
            )
            reverse_diffusion_sample(
                model,
                input_ids,
                s_cfg,
                attention_mask=attention_mask,
                observer=observer,
            )

            for (ti, layer, pool), rec in observer.records.items():
                target = target_ratios[ti]
                # pooled is (B, H); B == 1 for this loop.
                for b in range(rec.pooled.shape[0]):
                    meta = {
                        "prompt_id": prompt_id,
                        "source": row.get("source"),
                        "intended_group": row.get("intended_group"),
                        "actual_group": row.get("actual_group"),
                        "subconcept": row.get("subconcept"),
                        "length_bucket": row.get("length_bucket"),
                        "seed": seed,
                        "target_ratio": target,
                        "actual_ratio": float(rec.actual_ratio[b]),
                        "layer": layer,
                        "pool": pool,
                        "global_step": rec.global_step,
                        "block_index": rec.block_index,
                    }
                    writers[(ti, layer, pool)].append(rec.pooled[b], meta)
                    if token_writers and rec.token_level is not None:
                        token_writers[(ti, layer, pool)].append(
                            rec.token_level[b], meta
                        )
        print(f"  done prompt {prompt_id}", file=sys.stderr)

    for w in writers.values():
        w.close()
    for w in token_writers.values():
        w.close()
    print(f"Activations written under {args.out_root / 'activations'}")


if __name__ == "__main__":
    main()
