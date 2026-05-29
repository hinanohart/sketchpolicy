"""The augmentation spine: multiply one feasible plan into N feasible variants.

The loop draws transform parameters from a seeded Sobol sequence, applies the
transform, checks feasibility and keeps the feasible variants until ``n`` are
collected (reject-and-resample). Because both the Sobol sequence and the
rejection test are deterministic, the whole multiplication is reproducible: the
same ``(plan, n, seed, profile, ranges)`` yields bit-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sketchpolicy.augment import feasibility, transforms
from sketchpolicy.augment.sampling import SobolSampler
from sketchpolicy.augment.transforms import ParamRanges
from sketchpolicy.eeplan import EEPlan
from sketchpolicy.profiles import RobotProfile, get_profile

#: How many candidate draws, as a multiple of the requested count, before we
#: conclude the feasible region is too small and raise.
_MAX_DRAW_FACTOR = 64


class FeasibilityExhausted(RuntimeError):
    """Raised when too few feasible variants can be found for the request.

    Usually means the source trajectory sits near the workspace boundary, so
    almost every perturbation leaves the envelope.
    """


@dataclass(frozen=True)
class AugmentResult:
    """The outcome of a multiplication run."""

    variants: list[EEPlan]
    n_requested: int
    n_drawn: int
    n_rejected: int
    seed: int
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.n_drawn == 0 else len(self.variants) / self.n_drawn


def multiply(
    plan: EEPlan,
    n: int,
    *,
    seed: int = 0,
    profile: RobotProfile | str = "generic_tabletop",
    ranges: ParamRanges | None = None,
) -> AugmentResult:
    """Produce ``n`` feasible kinematic variants of ``plan``.

    Args:
        plan: a valid source :class:`EEPlan` (EE-action trajectory).
        n: number of feasible variants to produce.
        seed: Sobol seed; fixes the entire (deterministic) output.
        profile: a :class:`RobotProfile` or the name of a built-in one.
        ranges: optional override of the parameter box.

    Raises:
        ValueError: if ``n`` is not positive.
        FeasibilityExhausted: if fewer than ``n`` feasible variants are found
            within the draw budget.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    resolved: RobotProfile = (
        get_profile(profile) if isinstance(profile, str) else profile
    )
    ranges = ranges or ParamRanges()

    sampler = SobolSampler(ranges=ranges, seed=seed)
    variants: list[EEPlan] = []
    reject_reasons: dict[str, int] = {}
    n_drawn = 0
    n_rejected = 0
    batch = max(8, n)
    budget = _MAX_DRAW_FACTOR * n

    while len(variants) < n:
        for params in sampler.draw(batch):
            n_drawn += 1
            candidate = transforms.apply(plan, params)
            report = feasibility.check(candidate, resolved)
            if report.feasible:
                variants.append(candidate)
                if len(variants) == n:
                    break
            else:
                n_rejected += 1
                reason = report.first_reason() or "unknown"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        if len(variants) < n and n_drawn >= budget:
            raise FeasibilityExhausted(
                f"only {len(variants)}/{n} feasible variants after {n_drawn} draws "
                f"(budget {budget}); reject reasons={reject_reasons}. The source "
                f"trajectory may sit near the workspace boundary of profile "
                f"{resolved.name!r}."
            )

    return AugmentResult(
        variants=variants,
        n_requested=n,
        n_drawn=n_drawn,
        n_rejected=n_rejected,
        seed=seed,
        reject_reasons=reject_reasons,
    )
