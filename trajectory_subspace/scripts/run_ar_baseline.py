"""AR baseline scaffold — token-position binned activation collection.

This mirrors ``collect_activations.py`` but for a left-to-right AR model
(LLaMA-style). Instead of binning by response-masked fraction, it bins by
token position within the generated response.

The script is written so only two pieces depend on the AR model:

- ``_ar_generate_with_hiddens``: runs ``model.generate`` with
  ``output_hidden_states=True`` and returns the sequence + per-step hidden
  states.
- ``_position_bins``: maps token index to a bin index.

Everything else (pooling, cell-directory layout, shard writer, metadata
schema) is shared with the LLaDA pipeline, so downstream analysis and
intervention scripts work unchanged.

This file is a scaffold: the generation path is correct but has not been
exercised at full scale. It is here so the package presents a single
``collect → analyze → intervene`` interface for both LLaDA and AR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from trajectory_subspace.io_utils import ShardWriter, read_jsonl, yaml_load


def _load_ar_model(cfg: Dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m_cfg = cfg["model"]
    dtype = getattr(torch, m_cfg.get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["path"], trust_remote_code=m_cfg.get("trust_remote_code", False)
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            m_cfg["path"],
            trust_remote_code=m_cfg.get("trust_remote_code", False),
            torch_dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _ar_generate_with_hiddens(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    layers: Sequence[int],
) -> Tuple[torch.Tensor, Dict[int, List[torch.Tensor]]]:
    """Greedy AR decode, capturing hidden states per step per requested layer.

    Returns (sequence, {layer: [step_hiddens...]}) where each step hidden is
    shape (B, H) — the last-token hidden state at that layer after step.
    """
    per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    x = prompt_ids.clone()
    attn = attention_mask.clone()
    past = None
    for _ in range(gen_length):
        with torch.no_grad():
            out = model(
                input_ids=x if past is None else x[:, -1:],
                attention_mask=attn,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        for l in layers:
            # hidden_states has len = num_layers + 1.
            h = out.hidden_states[l + 1][:, -1, :].detach().cpu()
            per_layer[l].append(h)
        x = torch.cat([x, next_id], dim=1)
        attn = torch.cat(
            [attn, torch.ones_like(next_id, dtype=attn.dtype)], dim=1
        )
        if (next_id == tokenizer.eos_token_id).all():
            break
    return x, per_layer


def _position_bins(num_positions: int, num_bins: int) -> List[Tuple[int, int]]:
    if num_positions == 0 or num_bins == 0:
        return []
    bins: List[Tuple[int, int]] = []
    edges = np.linspace(0, num_positions, num_bins + 1, dtype=int).tolist()
    for i in range(num_bins):
        a, b = edges[i], edges[i + 1]
        if b > a:
            bins.append((a, b))
    return bins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--labeled-manifest", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--group", default="ar")
    p.add_argument("--num-bins", type=int, default=7,
                   help="Number of token-position bins (aligns with LLaDA's 7 ratios).")
    p.add_argument("--limit-prompts", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml_load(args.config)
    layers = list(cfg["capture"]["layers"])
    poolings = ["bin_mean"]  # AR baseline uses bin-mean pooling.
    gen_length = cfg["sampling"]["gen_length"]

    manifest = list(read_jsonl(args.labeled_manifest))
    if args.limit_prompts is not None:
        manifest = manifest[: args.limit_prompts]

    model, tokenizer = _load_ar_model(cfg)
    shard_max_rows = cfg["io"].get("shard_max_rows", 512)

    # Cell directories match the LLaDA layout: substitute bin_index for
    # target_ratio. Names use ``t=bin<ii>`` so downstream can still group by
    # "target" cleanly.
    writers = {}
    num_bins = args.num_bins
    for bi in range(num_bins):
        for l in layers:
            for p in poolings:
                cell_dir = (
                    args.out_root
                    / "activations"
                    / "pooled"
                    / args.group
                    / f"t=bin{bi:02d}_l={l:02d}_p={p}"
                )
                writers[(bi, l, p)] = ShardWriter(cell_dir, shard_max_rows=shard_max_rows)

    for row in manifest:
        rendered = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, tokenize=False
        ) if hasattr(tokenizer, "apply_chat_template") else row["prompt"]
        enc = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        prompt_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)

        seq, per_layer = _ar_generate_with_hiddens(
            model, tokenizer, prompt_ids, attn, gen_length, layers
        )
        produced = seq.shape[1] - prompt_ids.shape[1]
        bins = _position_bins(produced, num_bins)
        for bi, (a, b) in enumerate(bins):
            for l in layers:
                steps = per_layer[l][a:b]
                if not steps:
                    continue
                stacked = torch.stack(steps, dim=0).mean(dim=0)  # (B, H)
                for sample_i in range(stacked.shape[0]):
                    meta = {
                        "prompt_id": row["id"],
                        "source": row.get("source"),
                        "intended_group": row.get("intended_group"),
                        "actual_group": row.get("actual_group"),
                        "subconcept": row.get("subconcept"),
                        "length_bucket": row.get("length_bucket"),
                        "seed": 0,
                        "target_ratio": f"bin{bi:02d}",
                        "actual_ratio": (a, b),
                        "layer": l,
                        "pool": "bin_mean",
                    }
                    writers[(bi, l, "bin_mean")].append(stacked[sample_i], meta)
        print(f"  AR done {row['id']}", file=sys.stderr)

    for w in writers.values():
        w.close()
    print(f"AR baseline activations written under {args.out_root}/activations")


if __name__ == "__main__":
    main()
