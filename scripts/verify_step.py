#!/usr/bin/env python3
"""Per-step build verification for the sketchpolicy autonomous build.

Each step asserts concrete artifacts exist *and* runs the relevant test subset,
so a green result cannot be vacuous (no always-true checks, no empty matrices).
Usage::

    python scripts/verify_step.py S1
    python scripts/verify_step.py S1 --dry-run   # artifact checks only, no tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "sketchpolicy"


def _exists(rel: str) -> bool:
    ok = (ROOT / rel).exists()
    print(f"  [{'OK' if ok else 'MISSING'}] {rel}")
    return ok


def _run_tests(paths: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths]
    print(f"  $ pytest {' '.join(paths)}")
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def _cli_help() -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "sketchpolicy.cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = r.returncode == 0 and "augment" in r.stdout
    print(f"  [{'OK' if ok else 'FAIL'}] `sketchpolicy --help` lists commands")
    return ok


# (files that must exist, test paths to run)
STEPS: dict[str, tuple[list[str], list[str]]] = {
    "S0_5": (
        [
            "pyproject.toml",
            "LICENSE",
            "NOTICE",
            "README.md",
            ".gitignore",
            "src/sketchpolicy/__init__.py",
        ],
        [],
    ),
    "S1": (
        [
            "src/sketchpolicy/eeplan.py",
            "src/sketchpolicy/augment/pipeline.py",
            "src/sketchpolicy/emit/writer.py",
            "src/sketchpolicy/replay/pybullet_replay.py",
        ],
        [
            "tests/test_eeplan.py",
            "tests/test_augment_transforms.py",
            "tests/test_augment_feasibility.py",
            "tests/test_augment_pipeline.py",
            "tests/test_emit_roundtrip.py",
        ],
    ),
    "S2": (
        [
            "src/sketchpolicy/ingest/mock.py",
            "src/sketchpolicy/ingest/mediapipe_hands.py",
            "src/sketchpolicy/retarget/gripper.py",
            "src/sketchpolicy/retarget/pose.py",
        ],
        ["tests/test_retarget.py", "tests/test_ingest_mock.py"],
    ),
    "S3": (["src/sketchpolicy/cli.py"], []),
    "S4": (
        ["src/sketchpolicy/doctor.py", "src/sketchpolicy/profiles/__init__.py"],
        ["tests/test_doctor.py"],
    ),
    "S5": (["src/sketchpolicy/contract.py"], ["tests/test_contract.py"]),
    "S6": (["bench_results/v0.1.0a1.json", "scripts/measure.py"], []),
    "S7": (["README.md"], ["tests/test_readme.py"]),
}


def _check_bench() -> bool:
    """Non-vacuous S6 check: the bench JSON exists and its measurements are real
    (non-null operational results), and it carries the no-accuracy-claim note."""
    import json

    path = ROOT / "bench_results" / "v0.1.0a1.json"
    if not path.exists():
        print("  [MISSING] bench_results/v0.1.0a1.json")
        return False
    data = json.loads(path.read_text())
    m = data.get("measurements", {})
    ok = (
        m.get("augment_roundtrip_ok") is True
        and m.get("deterministic_bit_exact") is True
        and m.get("reject_resample_discards_infeasible", {}).get("ok") is True
        and "no policy" in data.get("claim_note", "").lower()
    )
    print(
        f"  [{'OK' if ok else 'FAIL'}] bench measurements are real + no-accuracy-claim note present"
    )
    return ok


def verify(step: str, dry_run: bool) -> bool:
    if step == "S3":
        files, tests = STEPS[step]
        return all(_exists(f) for f in files) and (dry_run or _cli_help())
    if step not in STEPS:
        print(f"unknown step {step!r}; known: {sorted(STEPS)}")
        return False
    files, tests = STEPS[step]
    files_ok = all(_exists(f) for f in files)
    if step == "S6":  # non-vacuous: validate bench content, not just file existence
        return files_ok and (dry_run or _check_bench())
    if dry_run or not tests:
        return files_ok
    return files_ok and _run_tests(tests)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ok = verify(args.step, args.dry_run)
    print(f"{args.step}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
