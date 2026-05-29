"""The augmentation spine: deterministic, simulator-free demonstration multiplier.

Public API::

    from sketchpolicy.augment import multiply, AugmentResult

``multiply(plan, n, seed=..., profile=...)`` returns ``n`` feasible kinematic
variants of an :class:`~sketchpolicy.eeplan.EEPlan`.
"""

from __future__ import annotations

# Leaf modules first (no intra-package cycles), then the orchestrator.
from sketchpolicy.augment.feasibility import FeasibilityReport, check
from sketchpolicy.augment.sampling import SobolSampler
from sketchpolicy.augment.transforms import ParamRanges, TransformParams, apply
from sketchpolicy.augment.pipeline import (
    AugmentResult,
    FeasibilityExhausted,
    multiply,
)

__all__ = [
    "multiply",
    "AugmentResult",
    "FeasibilityExhausted",
    "apply",
    "TransformParams",
    "ParamRanges",
    "SobolSampler",
    "check",
    "FeasibilityReport",
]
