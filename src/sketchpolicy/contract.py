"""Single source of truth for the project's honest-marketing contract.

Every claim, disclaimer and the runtime acknowledgement gate live here so that
the README, the CLI and the tests all read from one place. The honest-marketing
test asserts (with ``==`` exact-string comparison, never a shell grep) that no
banned overclaim phrase appears in the disclaimer, and a negative fixture proves
the banned-phrase scan actually fires.
"""

from __future__ import annotations

#: The explicit acknowledgement flag required by every *write* command.
BOOTSTRAP_FLAG = "--i-understand-this-is-a-bootstrap-scaffold"

#: Things sketchpolicy deliberately does NOT claim. Kept as data so the README
#: generator and tests can render/verify them verbatim.
NON_CLAIMS: tuple[str, ...] = (
    "Does NOT improve any policy success rate (data-shaping only; untested downstream).",
    "Does NOT recover absolute metric scale or full 6-DoF from monocular hand video.",
    "Does NOT perform sim-to-real and does NOT transform camera video.",
    "Augments the proprioceptive / action stream only (EE pose + gripper); v0.1 output is action/state-only.",
    "Parallel-jaw gripper only in v0.1 (no dexterous / bimanual).",
    "Uses MediaPipe Hands only for hand pose; does NOT use MANO / HaMeR / WiLoR.",
    "Feasibility is a kinematic-envelope check (reach/floor/box/base-keepout); it does NOT test robot-mesh self-collision.",
)

#: The disclaimer shown at the top of every write command and embedded in the
#: README. Phrased so that it can never read as an over-promise.
DISCLAIMER = (
    "sketchpolicy is a pre-alpha bootstrap scaffold. It produces schema-valid "
    "LeRobotDataset v3.0 episodes by deterministic kinematic transforms and a "
    "feasibility filter. It does not improve policy performance, does not do "
    "sim-to-real, and does not recover absolute scale from monocular video. "
    "You are responsible for validating that any generated data is appropriate "
    "for your downstream use."
)

#: Phrases that must never appear in user-facing marketing copy. The
#: honest-marketing test scans the disclaimer and README for these (exact,
#: case-insensitive substring) and fails the build if any is present. These are
#: intentionally listed only here (CONTRIBUTING references this module by name)
#: so the scan never matches its own definition in a scanned file.
BANNED_OVERCLAIM_PHRASES: tuple[str, ...] = (
    "state-of-the-art",
    "production-ready",
    "guaranteed to improve",
    "sim-to-real ready",
    "improves success rate",
    "fully automatic",
    "permanent",
    "solves robot learning",
)


class BootstrapAckError(RuntimeError):
    """Raised when a write command is invoked without the acknowledgement flag."""


def require_bootstrap_ack(acknowledged: bool) -> None:
    """Gate a write command on the explicit bootstrap acknowledgement.

    Args:
        acknowledged: whether ``BOOTSTRAP_FLAG`` was passed on the command line.

    Raises:
        BootstrapAckError: if ``acknowledged`` is false.
    """
    if not acknowledged:
        raise BootstrapAckError(
            f"This command writes data and requires the flag {BOOTSTRAP_FLAG!r}. "
            f"{DISCLAIMER}"
        )


def scan_for_banned_phrases(text: str) -> list[str]:
    """Return the banned overclaim phrases found in ``text`` (case-insensitive).

    Used by the honest-marketing test. Returns an empty list for clean copy.
    """
    low = text.lower()
    return [p for p in BANNED_OVERCLAIM_PHRASES if p.lower() in low]
