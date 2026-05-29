"""Command-line interface for sketchpolicy.

Subcommands:

* ``augment`` - multiply an EE-action LeRobotDataset v3.0 into feasible variants;
* ``sketch``  - (experimental) turn a hand video into one EE episode;
* ``replay``  - (read-only) headless kinematic sanity replay of one episode;
* ``doctor``  - report available capabilities.

Write commands (``augment``, ``sketch``) require the explicit acknowledgement
flag ``--i-understand-this-is-a-bootstrap-scaffold``.
"""

from __future__ import annotations

import argparse
import sys

from sketchpolicy import __version__
from sketchpolicy.contract import (
    BOOTSTRAP_FLAG,
    DISCLAIMER,
    BootstrapAckError,
    require_bootstrap_ack,
)
from sketchpolicy.profiles import available_profiles

_ACK_DEST = "i_understand_this_is_a_bootstrap_scaffold"


def _add_ack_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        BOOTSTRAP_FLAG, dest=_ACK_DEST, action="store_true", help=argparse.SUPPRESS
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sketchpolicy",
        description="Simulator-free, CPU-only kinematic demonstration multiplier "
        "for LeRobotDataset v3.0 (pre-alpha bootstrap scaffold).",
    )
    parser.add_argument(
        "--version", action="version", version=f"sketchpolicy {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser(
        "augment", help="multiply an EE-action dataset into feasible variants"
    )
    pa.add_argument("dataset", help="path to a source LeRobotDataset v3.0 (EE-action)")
    pa.add_argument("--out", required=True, help="output dataset directory")
    pa.add_argument(
        "--n", type=int, default=10, help="feasible variants per source episode"
    )
    pa.add_argument("--seed", type=int, default=0, help="Sobol seed (fixes the output)")
    pa.add_argument(
        "--profile",
        default="generic_tabletop",
        choices=available_profiles(),
        help="robot workspace profile",
    )
    pa.add_argument(
        "--include-source",
        dest="include_source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the original episodes alongside the variants (default: yes)",
    )
    pa.add_argument(
        "--overwrite", action="store_true", help="replace an existing output dir"
    )
    _add_ack_flag(pa)
    pa.set_defaults(func=cmd_augment)

    ps = sub.add_parser("sketch", help="(experimental) hand video -> one EE episode")
    ps.add_argument("video", help="path to a monocular hand video")
    ps.add_argument("--out", required=True, help="output dataset directory")
    ps.add_argument("--model", default=None, help="path to hand_landmarker.task")
    ps.add_argument(
        "--z-mode",
        default="constant",
        choices=["constant", "depth_heuristic"],
        dest="z_mode",
        help="planar (default) or experimental depth heuristic",
    )
    ps.add_argument("--overwrite", action="store_true")
    _add_ack_flag(ps)
    ps.set_defaults(func=cmd_sketch)

    pr = sub.add_parser("replay", help="(read-only) headless kinematic sanity replay")
    pr.add_argument("dataset", help="path to a LeRobotDataset v3.0")
    pr.add_argument("--episode", type=int, default=0, help="episode index to replay")
    pr.set_defaults(func=cmd_replay)

    pd = sub.add_parser("doctor", help="report available capabilities")
    pd.set_defaults(func=cmd_doctor)

    return parser


def _print_disclaimer() -> None:
    print(f"[sketchpolicy] {DISCLAIMER}", file=sys.stderr)


def cmd_augment(args: argparse.Namespace) -> int:
    from sketchpolicy.augment import FeasibilityExhausted, multiply
    from sketchpolicy.emit import read_plans, write_dataset

    _print_disclaimer()
    require_bootstrap_ack(getattr(args, _ACK_DEST))

    sources = read_plans(args.dataset)
    out_plans = []
    total_rejected = 0
    for ep_idx, plan in enumerate(sources):
        if args.include_source:
            out_plans.append(plan)
        try:
            # Offset the Sobol seed per source episode so a multi-episode input
            # does not receive the identical transform-parameter sequence for
            # every episode. Still fully deterministic in (seed, episode order);
            # a single-episode input (ep_idx == 0) is bit-identical to before.
            res = multiply(
                plan, n=args.n, seed=args.seed + ep_idx, profile=args.profile
            )
        except FeasibilityExhausted as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out_plans.extend(res.variants)
        total_rejected += res.n_rejected

    write_dataset(out_plans, args.out, overwrite=args.overwrite)
    print(
        f"augmented {len(sources)} source episode(s) -> {len(out_plans)} episodes "
        f"({args.n} variants each, {total_rejected} infeasible rejected) at {args.out}"
    )
    return 0


def cmd_sketch(args: argparse.Namespace) -> int:
    from sketchpolicy.emit import write_dataset
    from sketchpolicy.ingest import IngestUnavailable, ingest_video
    from sketchpolicy.retarget import RetargetConfig, retarget_track

    _print_disclaimer()
    require_bootstrap_ack(getattr(args, _ACK_DEST))
    try:
        track = ingest_video(args.video, model_path=args.model)
    except IngestUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    plan = retarget_track(track, RetargetConfig(z_mode=args.z_mode))
    write_dataset([plan], args.out, overwrite=args.overwrite)
    print(f"sketched 1 experimental episode ({len(plan)} frames) at {args.out}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from sketchpolicy.emit import read_plans
    from sketchpolicy.replay import ReplayUnavailable, replay_plan

    plans = read_plans(args.dataset)
    if not 0 <= args.episode < len(plans):
        print(
            f"error: episode {args.episode} out of range [0, {len(plans)})",
            file=sys.stderr,
        )
        return 1
    try:
        report = replay_plan(plans[args.episode])
    except ReplayUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"replay episode {args.episode}: {report.n_frames} frames, "
        f"max residual {report.max_position_residual_m:.4f} m, "
        f"reachable {report.reachable_fraction:.0%} (threshold {report.threshold_m} m)"
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from sketchpolicy.doctor import format_report, run_doctor

    print(format_report(run_doctor()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BootstrapAckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
