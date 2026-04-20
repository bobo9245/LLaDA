"""LLaDA reverse-diffusion sampler (backwards-compatible entry point).

The actual sampling loop lives in ``trajectory_subspace.sampling``. This module
keeps the original public surface (``generate``, ``add_gumbel_noise``,
``get_num_transfer_tokens``) so ``chat.py``, ``eval_llada.py``, the
OpenCompass wrapper, and any third-party callers continue to work unchanged.
"""

import os
import sys

import torch
from transformers import AutoTokenizer, AutoModel

# Make the ``trajectory_subspace`` package importable when this file is used
# from a working directory other than the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from trajectory_subspace.sampling import (  # noqa: E402
    add_gumbel_noise,
    get_num_transfer_tokens,
    sample_like_generate,
)


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
):
    """Reverse-diffusion sample. See ``trajectory_subspace.sampling`` for docs.

    The signature is preserved exactly to keep ``eval_llada.py`` and the
    OpenCompass wrapper bit-compatible.
    """
    return sample_like_generate(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )


def main():
    device = "cuda"

    model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != 126336

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    ]

    messages = [{"role": "user", "content": p} for p in prompts]
    prompts = [
        tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
        for m in messages
    ]

    encoded_outputs = tokenizer(
        prompts, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    input_ids = encoded_outputs["input_ids"].to(device)
    attention_mask = encoded_outputs["attention_mask"].to(device)

    out = generate(
        model,
        input_ids,
        attention_mask,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    output = tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)
    for o in output:
        print(o)
        print("-" * 50)


__all__ = ["generate", "add_gumbel_noise", "get_num_transfer_tokens", "main"]


if __name__ == "__main__":
    main()
