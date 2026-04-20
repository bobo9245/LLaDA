"""Storage helpers: shard activation tensors and manage JSONL manifests.

The on-disk layout produced by :mod:`scripts.collect_activations` is:

::

    <out_root>/
      manifests/<group>.jsonl
      generations/<group>.jsonl
      activations/pooled/<group>/<cell>/shard-0000.safetensors
      activations/pooled/<group>/<cell>/index.jsonl
      activations/token_level/<group>/<cell>/...
      activations/gradient/<group>/<cell>/...

Each ``shard-*.safetensors`` carries a batch of pooled activations along with
aligned per-row metadata columns. ``index.jsonl`` records row->shard mapping
so resuming a partial run is straightforward.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch

try:
    from safetensors.torch import load_file as _safe_load
    from safetensors.torch import save_file as _safe_save
except ImportError:  # pragma: no cover
    _safe_load = None
    _safe_save = None


def _require_safetensors() -> None:
    if _safe_save is None:
        raise ImportError(
            "safetensors is required for activation storage. "
            "Install with `pip install safetensors`."
        )


def read_jsonl(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: os.PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: os.PathLike, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class ShardWriter:
    """Buffered writer that flushes ``shard_max_rows`` rows per file.

    ``meta`` is a list of dicts, one per row, written to ``index.jsonl``
    alongside the safetensors file. The dict must include the fields caller
    wants to filter on downstream (e.g. ``prompt_id``, ``seed``,
    ``target_ratio``, ``layer``, ``pool``, ``intended_group``).
    """

    out_dir: os.PathLike
    shard_max_rows: int = 512
    _buffer: List[torch.Tensor] = None
    _meta: List[Dict[str, Any]] = None
    _shard_idx: int = 0
    _row_cursor: int = 0

    def __post_init__(self) -> None:
        _require_safetensors()
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = []
        self._meta = []
        # Resume: advance shard_idx past any existing shards so we do not
        # overwrite them.
        existing = sorted(self.out_dir.glob("shard-*.safetensors"))
        self._shard_idx = len(existing)
        index_path = self.out_dir / "index.jsonl"
        if index_path.exists():
            self._row_cursor = sum(1 for _ in read_jsonl(index_path))

    def already_has(self, predicate) -> bool:
        """Return True iff any existing indexed row matches ``predicate``."""
        index_path = self.out_dir / "index.jsonl"
        if not index_path.exists():
            return False
        for row in read_jsonl(index_path):
            if predicate(row):
                return True
        return False

    def append(self, tensor: torch.Tensor, meta: Mapping[str, Any]) -> None:
        self._buffer.append(tensor.detach().to("cpu"))
        self._meta.append(dict(meta))
        if len(self._buffer) >= self.shard_max_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        stacked = torch.stack(self._buffer, dim=0) if self._buffer[0].dim() > 0 else torch.tensor(self._buffer)
        shard_path = self.out_dir / f"shard-{self._shard_idx:04d}.safetensors"
        _safe_save({"values": stacked}, str(shard_path))
        index_path = self.out_dir / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as fh:
            for offset, m in enumerate(self._meta):
                row = {
                    "shard": shard_path.name,
                    "offset": offset,
                    "row": self._row_cursor + offset,
                    **m,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._row_cursor += len(self._buffer)
        self._shard_idx += 1
        self._buffer = []
        self._meta = []

    def close(self) -> None:
        self.flush()


def load_cell(cell_dir: os.PathLike) -> Dict[str, Any]:
    """Load every shard in ``cell_dir`` and concatenate into a single tensor."""
    _require_safetensors()
    cell_dir = Path(cell_dir)
    index_path = cell_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.jsonl at {cell_dir}")
    rows = list(read_jsonl(index_path))
    shards: Dict[str, torch.Tensor] = {}
    for shard_name in sorted({r["shard"] for r in rows}):
        shards[shard_name] = _safe_load(str(cell_dir / shard_name))["values"]
    values = torch.cat(
        [shards[r["shard"]][r["offset"]].unsqueeze(0) for r in rows], dim=0
    )
    return {"values": values, "meta": rows}


def yaml_load(path: os.PathLike) -> Dict[str, Any]:
    """Tiny YAML loader: uses PyYAML if available, else returns ``json.load``.

    Config files in this package are intentionally simple — the subset we use
    is a strict superset of JSON so falling back to JSON works when PyYAML is
    unavailable.
    """
    path = Path(path)
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def yaml_dump(obj: Any, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(obj, fh, sort_keys=False)
    except ImportError:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
