"""Torch-free LeRobotDataset v3.0 emission (the P3 hand-rolled writer path).

Public API::

    from sketchpolicy.emit import write_dataset, read_plans, validate_dataset
"""

from __future__ import annotations

from sketchpolicy.emit.reader import SchemaError, read_plans, validate_dataset
from sketchpolicy.emit.writer import EmitError, write_dataset

__all__ = [
    "write_dataset",
    "read_plans",
    "validate_dataset",
    "EmitError",
    "SchemaError",
]
