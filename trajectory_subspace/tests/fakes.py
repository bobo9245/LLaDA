"""Test doubles for the sampling pipeline.

We want tests to run on CPU, without the real LLaDA weights. The sampler only
needs a module that (a) has a ``device`` attribute, (b) accepts
``(input_ids, attention_mask=None, output_hidden_states=?)`` and (c) returns
an object with ``.logits`` (and optionally ``.hidden_states``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


class _FakeBlock(torch.nn.Module):
    """Block that meets the block-discovery heuristic (>=2 children)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        # Keep this close to identity so hidden states are stable.
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(hidden_size))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.eye(hidden_size))
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


@dataclass
class _Out:
    logits: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None


class FakeLLaDAModel(torch.nn.Module):
    """Deterministic tiny "LLaDA" that always picks the same token.

    Token dynamics: ``argmax`` is always token id ``fill_token`` (which we set
    to ``0`` by default). Confidence is roughly uniform, so the sampler will
    unmask positions in the topk order dictated by positional biases that we
    craft in ``logits``.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 16,
        num_layers: int = 3,
        fill_token: int = 0,
        supports_output_hidden_states: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fill_token = fill_token
        self.supports_output_hidden_states = supports_output_hidden_states
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.blocks = torch.nn.ModuleList(
            [_FakeBlock(hidden_size) for _ in range(num_layers)]
        )
        self.out = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            # Make ``fill_token`` the argmax everywhere.
            self.out.weight.zero_()
            self.out.weight[fill_token] = 1.0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> _Out:
        if output_hidden_states and not self.supports_output_hidden_states:
            # Mimic remote-code models that don't accept the kwarg.
            raise TypeError("output_hidden_states not supported")
        h = self.embed(input_ids)
        hiddens: List[torch.Tensor] = []
        if output_hidden_states:
            hiddens.append(h)
        for block in self.blocks:
            h = block(h)
            if output_hidden_states:
                hiddens.append(h)
        logits = self.out(h)
        return _Out(
            logits=logits,
            hidden_states=tuple(hiddens) if output_hidden_states else None,
        )


class NoHiddenFakeLLaDAModel(FakeLLaDAModel):
    """Fake model that refuses ``output_hidden_states`` — exercises hook fallback."""

    def __init__(self, **kwargs):
        super().__init__(supports_output_hidden_states=False, **kwargs)
