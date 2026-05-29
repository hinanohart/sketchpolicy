"""CLI tests via direct main() invocation (no subprocess)."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.cli import main
from sketchpolicy.eeplan import EEPlan
from sketchpolicy.emit import read_plans, write_dataset


def _write_source(path) -> None:
    t = np.linspace(0.0, 1.0, 20)
    pos = np.stack(
        [0.40 + 0.10 * np.cos(t * np.pi), 0.10 * np.sin(t * np.pi), 0.20 + 0.05 * t],
        axis=1,
    )
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (20, 1))
    g = np.clip(0.5 + 0.4 * np.sin(t * 6), 0.0, 1.0)
    write_dataset(
        [
            EEPlan(
                positions=pos,
                quaternions=q,
                gripper=g,
                timestamps=t,
                fps=30.0,
                task="pick",
            )
        ],
        path,
    )


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sketchpolicy" in capsys.readouterr().out


def test_doctor_returns_zero() -> None:
    assert main(["doctor"]) == 0


def test_augment_requires_ack(tmp_path) -> None:
    src = tmp_path / "src"
    _write_source(src)
    rc = main(["augment", str(src), "--out", str(tmp_path / "out"), "--n", "3"])
    assert rc == 2  # bootstrap gate
    assert not (tmp_path / "out").exists()


def test_augment_with_ack_writes_dataset(tmp_path) -> None:
    src = tmp_path / "src"
    _write_source(src)
    out = tmp_path / "out"
    rc = main(
        [
            "augment",
            str(src),
            "--out",
            str(out),
            "--n",
            "4",
            "--seed",
            "0",
            "--i-understand-this-is-a-bootstrap-scaffold",
        ]
    )
    assert rc == 0
    plans = read_plans(out)
    assert len(plans) == 5  # 1 source + 4 variants


def test_augment_no_include_source(tmp_path) -> None:
    src = tmp_path / "src"
    _write_source(src)
    out = tmp_path / "out"
    rc = main(
        [
            "augment",
            str(src),
            "--out",
            str(out),
            "--n",
            "4",
            "--no-include-source",
            "--i-understand-this-is-a-bootstrap-scaffold",
        ]
    )
    assert rc == 0
    assert len(read_plans(out)) == 4  # variants only


def test_replay_out_of_range(tmp_path) -> None:
    src = tmp_path / "src"
    _write_source(src)
    assert main(["replay", str(src), "--episode", "99"]) == 1
