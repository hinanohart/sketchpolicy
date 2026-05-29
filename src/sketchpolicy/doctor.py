"""Environment / dependency self-report (``sketchpolicy doctor``).

Honestly reports which optional capabilities are available so a user knows,
before running, whether the experimental sketch path or the replay path will
work on their machine — including the easy-to-miss system GL libraries that
MediaPipe Tasks needs at runtime.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
from dataclasses import dataclass, field

from sketchpolicy.profiles import available_profiles

# (import name, human label, which extra provides it)
_CORE = [
    ("numpy", "numpy", "core"),
    ("scipy", "scipy", "core"),
    ("pyarrow", "pyarrow", "core"),
]
_OPTIONAL = [
    ("mediapipe", "mediapipe", "ingest"),
    ("cv2", "opencv-python-headless", "ingest"),
    ("av", "PyAV", "ingest"),
    ("pybullet", "pybullet", "replay"),
    ("ledataset", "lerobot-dataset", "lerobot"),
]


@dataclass
class DoctorReport:
    components: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def capability(self, extra: str) -> bool:
        """True if every component of an extra is importable."""
        members = [c for c in self.components.values() if c["extra"] == extra]
        return bool(members) and all(c["present"] for c in members)


def _probe(import_name: str, label: str, extra: str) -> dict:
    info = {
        "label": label,
        "extra": extra,
        "present": False,
        "version": None,
        "note": "",
    }
    try:
        # `import_name` only ever comes from the fixed _CORE/_OPTIONAL whitelist
        # below, never from user input -- the whitelist mitigation semgrep asks for.
        importlib.import_module(import_name)  # nosemgrep
        info["present"] = True
        try:
            info["version"] = md.version(label)
        except md.PackageNotFoundError:
            info["version"] = "?"
    except OSError as exc:
        # importable package but a runtime shared library is missing (e.g. GL).
        info["note"] = f"present but failed to load: {exc}"
    except ImportError:
        info["note"] = f"not installed (pip install 'sketchpolicy[{extra}]')"
    return info


def run_doctor() -> DoctorReport:
    report = DoctorReport()
    for import_name, label, extra in _CORE + _OPTIONAL:
        report.components[label] = _probe(import_name, label, extra)
    if not report.capability("ingest"):
        report.notes.append(
            "Experimental 'sketch' path unavailable. It needs `pip install "
            "'sketchpolicy[ingest]'` and, on Linux, the system GL libraries "
            "(e.g. libgles2, libegl1) that MediaPipe Tasks loads at runtime."
        )
    if not report.capability("replay"):
        report.notes.append(
            "'replay' path unavailable: `pip install 'sketchpolicy[replay]'`."
        )
    return report


def format_report(report: DoctorReport) -> str:
    lines = [
        "sketchpolicy doctor",
        "=" * 40,
        "",
        "Profiles: " + ", ".join(available_profiles()),
        "",
    ]
    for label, info in report.components.items():
        mark = "OK " if info["present"] else "-- "
        ver = f" {info['version']}" if info["version"] else ""
        extra = "" if info["extra"] == "core" else f" [{info['extra']}]"
        note = f"  ({info['note']})" if info["note"] else ""
        lines.append(f"  [{mark}] {label}{ver}{extra}{note}")
    if report.notes:
        lines.append("")
        lines.extend(f"NOTE: {n}" for n in report.notes)
    return "\n".join(lines)
