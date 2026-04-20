"""LLaDA trajectory-subspace experiment package.

Public entry points for library consumers:

- ``SamplingConfig`` / ``reverse_diffusion_sample`` — shared sampling core.
- ``TrajectoryObserver`` / ``NullObserver`` — step-level hooks fed by the sampler.
- ``Intervention`` / ``NullIntervention`` — hidden-state edits applied during forward.
- ``CollectObserver`` — an observer that captures activations keyed by
  response-masked fraction.

Scripts live under ``trajectory_subspace.scripts.*`` and are callable as
``python -m trajectory_subspace.scripts.<name>``.
"""

from .sampling import (
    SamplingConfig,
    SamplingState,
    ForwardOutput,
    TrajectoryObserver,
    NullObserver,
    Intervention,
    NullIntervention,
    reverse_diffusion_sample,
)
from .observers import CollectObserver
from .interventions import (
    ProjectionAblation,
    Steering,
    RandomSubspaceControl,
)

__all__ = [
    "SamplingConfig",
    "SamplingState",
    "ForwardOutput",
    "TrajectoryObserver",
    "NullObserver",
    "Intervention",
    "NullIntervention",
    "reverse_diffusion_sample",
    "CollectObserver",
    "ProjectionAblation",
    "Steering",
    "RandomSubspaceControl",
]
