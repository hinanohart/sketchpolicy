#!/usr/bin/env python3
"""Produce bench_results/v<version>.json from live measurements.

These are *operational* metrics only — round-trip validity, determinism, the
feasibility filter, MediaPipe CPU throughput and a replay smoke test. No
accuracy or policy-success metric is produced or implied: there is no robot
ground truth here, and claiming one would be dishonest.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sketchpolicy import __version__
from sketchpolicy.augment import multiply
from sketchpolicy.eeplan import EEPlan
from sketchpolicy.emit import read_plans, write_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench_results" / f"v{__version__}.json"


def _source_plan() -> EEPlan:
    t = np.linspace(0.0, 1.0, 30)
    pos = np.stack(
        [0.40 + 0.10 * np.cos(t * np.pi), 0.10 * np.sin(t * np.pi), 0.20 + 0.05 * t],
        axis=1,
    )
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (30, 1))
    g = np.clip(0.5 + 0.4 * np.sin(t * 2 * np.pi), 0.0, 1.0)
    return EEPlan(
        positions=pos, quaternions=q, gripper=g, timestamps=t, fps=30.0, task="pick"
    )


def _boundary_plan() -> EEPlan:
    t = np.linspace(0.0, 1.0, 24)
    r = 0.78
    cx, cy = r * np.cos(np.pi / 4), r * np.sin(np.pi / 4)
    pos = np.stack(
        [cx + 0.01 * np.cos(t * np.pi), cy + 0.01 * np.sin(t * np.pi), 0.12 + 0.02 * t],
        axis=1,
    )
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (24, 1))
    return EEPlan(
        positions=pos, quaternions=q, gripper=np.full(24, 0.5), timestamps=t, fps=30.0
    )


def measure_roundtrip() -> bool:
    plan = _source_plan()
    variants = multiply(plan, n=5, seed=0).variants
    with tempfile.TemporaryDirectory() as d:
        out = write_dataset([plan, *variants], Path(d) / "ds")
        back = read_plans(out)
    return len(back) == 6 and all(
        np.allclose(a.to_action_array(), b.to_action_array(), atol=1e-6)
        for a, b in zip([plan, *variants], back)
    )


def measure_determinism() -> bool:
    plan = _source_plan()
    a = multiply(plan, n=8, seed=7).variants
    b = multiply(plan, n=8, seed=7).variants
    return all(
        np.array_equal(x.to_action_array(), y.to_action_array()) for x, y in zip(a, b)
    )


def measure_reject_resample() -> dict:
    res = multiply(_boundary_plan(), n=5, seed=0)
    return {
        "ok": len(res.variants) == 5 and res.n_rejected >= 1,
        "n_drawn": res.n_drawn,
        "n_rejected": res.n_rejected,
        "reject_reasons": res.reject_reasons,
    }


def measure_mediapipe_fps() -> dict:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from sketchpolicy.ingest.mediapipe_hands import ensure_model

        model = ensure_model()
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        lm = vision.HandLandmarker.create_from_options(opts)
        rng = np.random.default_rng(0)
        frames = [
            rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(64)
        ]
        lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[0]), 0
        )
        # Median of 3 passes: per-frame CPU inference time is noisy under load,
        # so a single sample is not representative (a reviewer flagged this).
        ts = 0
        fps_samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            for i, f in enumerate(frames):
                lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=f),
                    int(ts + (i + 1) * 33),
                )
            fps_samples.append(64 / (time.perf_counter() - t0))
            ts += 64 * 33
        lm.close()
        return {
            "available": True,
            "fps": round(float(np.median(fps_samples)), 1),
            "fps_samples": [round(s, 1) for s in fps_samples],
            "frames": 64,
            "resolution": "480x640",
            "mode": "synthetic-frames-throughput-median-of-3",
            "note": "Median CPU inference throughput on synthetic frames; representative, not a detection-accuracy metric and machine-dependent.",
        }
    except Exception as exc:  # ImportError / OSError(GL) / model missing
        return {"available": False, "fps": None, "reason": str(exc)[:200]}


def measure_replay() -> dict:
    try:
        from sketchpolicy.replay import replay_plan

        t = np.linspace(0.0, 1.0, 12)
        pos = np.stack(
            [
                0.40 + 0.08 * np.cos(t * np.pi),
                0.08 * np.sin(t * np.pi),
                0.50 + 0.05 * t,
            ],
            axis=1,
        )
        q = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (12, 1))
        plan = EEPlan(
            positions=pos,
            quaternions=q,
            gripper=np.full(12, 0.5),
            timestamps=t,
            fps=30.0,
        )
        rep = replay_plan(plan)
        return {
            "available": True,
            "ok": np.isfinite(rep.max_position_residual_m),
            "n_frames": rep.n_frames,
            "max_residual_m": round(rep.max_position_residual_m, 4),
            "reachable_fraction": rep.reachable_fraction,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def main() -> int:
    results = {
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "pyarrow": _pkg_version("pyarrow"),
        },
        "measurements": {
            "augment_roundtrip_ok": measure_roundtrip(),
            "deterministic_bit_exact": measure_determinism(),
            "reject_resample_discards_infeasible": measure_reject_resample(),
            "mediapipe_cpu_fps": measure_mediapipe_fps(),
            "pybullet_replay_smoke": measure_replay(),
        },
        "claim_note": (
            "Operational metrics only. No policy success-rate, sim-to-real or "
            "accuracy claim is made or measurable here (there is no robot ground "
            "truth in this benchmark)."
        ),
    }

    def _coerce(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, default=_coerce)
    OUT.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
