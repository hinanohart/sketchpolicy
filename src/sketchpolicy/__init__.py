"""sketchpolicy: simulator-free, CPU-only kinematic demonstration multiplier
for LeRobotDataset v3.0.

The public surface intentionally re-exports only the stable type boundary
(:mod:`sketchpolicy.eeplan`) and the contract constants. Heavier modules
(augment / emit / replay / ingest) are imported lazily by the CLI so that the
core install stays torch-free and importable without the optional extras.
"""

from sketchpolicy.contract import (
    BOOTSTRAP_FLAG,
    DISCLAIMER,
    NON_CLAIMS,
    BootstrapAckError,
    require_bootstrap_ack,
)
from sketchpolicy.eeplan import EEPlan, EEStep

__version__ = "0.1.0a3"

__all__ = [
    "__version__",
    "EEPlan",
    "EEStep",
    "BOOTSTRAP_FLAG",
    "DISCLAIMER",
    "NON_CLAIMS",
    "BootstrapAckError",
    "require_bootstrap_ack",
]
