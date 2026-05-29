"""Low-discrepancy sampling of the augmentation parameter space.

A scrambled Sobol sequence covers the parameter box far more evenly than i.i.d.
uniform draws, which matters because we then *reject* infeasible points: good
space-filling means fewer draws are wasted. The engine is stateful, so repeated
``draw`` calls continue the same sequence — that is what makes reject-and-resample
both efficient and deterministic for a fixed seed.
"""

from __future__ import annotations

import inspect
import warnings

from scipy.stats import qmc

from sketchpolicy.augment.transforms import N_PARAMS, ParamRanges, TransformParams

# scipy renamed the seeding kwarg from ``seed`` (<=1.14) to ``rng`` (>=1.15,
# with ``seed`` kept as a deprecated alias). Detect the canonical name once so
# we neither break on old scipy nor emit a DeprecationWarning on new scipy.
_SOBOL_SEED_KW = (
    "rng" if "rng" in inspect.signature(qmc.Sobol.__init__).parameters else "seed"
)


def _make_sobol(d: int, seed: int) -> qmc.Sobol:
    if _SOBOL_SEED_KW == "rng":
        return qmc.Sobol(d=d, scramble=True, rng=seed)
    return qmc.Sobol(d=d, scramble=True, seed=seed)  # type: ignore[call-arg]  # old scipy


class SobolSampler:
    """A seeded, scrambled Sobol sampler over the transform parameter box."""

    def __init__(self, ranges: ParamRanges | None = None, seed: int = 0) -> None:
        self.ranges = ranges or ParamRanges()
        self.seed = int(seed)
        self._engine = _make_sobol(N_PARAMS, self.seed)

    def draw(self, m: int) -> list[TransformParams]:
        """Draw ``m`` parameter points, continuing the Sobol sequence."""
        if m <= 0:
            return []
        with warnings.catch_warnings():
            # Sobol warns when m is not a power of two; balance is still fine
            # for our reject-and-resample use and the sequence stays deterministic.
            warnings.simplefilter("ignore", category=UserWarning)
            u = self._engine.random(m)
        scaled = self.ranges.scale_unit_cube(u)
        return [TransformParams.from_vector(scaled[i]) for i in range(m)]
