"""PLATO command line interface.

    plato init      write a starter config
    plato index     parse plate map + filenames, build the index, validate
    plato thumbs    build the thumbnail cache
    plato gui       open the browser
    plato check     re-print the last validation report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, load_config, write_template


def _config(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def cmd_init(args: argparse.Namespace) -> int:
    path = write_template(args.path)
    print(f"wrote {path}\nEdit [images].pattern and [platemap] to match your data, then run:")
    print(f"  plato index -c {path}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .data.index import build_index

    cfg = _config(args)
    report = build_index(cfg)
    if not report.ok:
        print(
            "\nThe index was built, but the mismatches above are worth reading before "
            "you trust anything you see in the browser.",
            file=sys.stderr,
        )
        return 2 if args.strict else 0
    return 0


def cmd_thumbs(args: argparse.Namespace) -> int:
    from .cache import build_thumbnails

    cfg = _config(args)
    build_thumbnails(
        cfg.db_path,
        cfg.thumb_db_path,
        size=cfg.thumbnails.size,
        percentiles=cfg.thumbnails.percentiles,
        sample_size=cfg.thumbnails.sample_size,
        workers=args.workers,
        force=args.force,
        autoscale=not args.no_autoscale,
    )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import run

    cfg = None
    if args.config.exists():
        candidate = _config(args)
        # report_path only exists after a successful 'plato index' run, so this
        # tells apart a real index from a stray/partial .plato/index.sqlite.
        if candidate.report_path.exists():
            cfg = candidate
    return run(cfg)


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if not cfg.report_path.exists():
        print(f"no report at {cfg.report_path}; run 'plato index' first")
        return 1
    print(json.dumps(json.loads(cfg.report_path.read_text()), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plato", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a starter config file")
    p_init.add_argument("path", nargs="?", type=Path, default=Path("plato.toml"))
    p_init.set_defaults(func=cmd_init)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", type=Path, default=Path("plato.toml"))

    p_index = sub.add_parser("index", help="build the index and validate the join")
    add_config(p_index)
    p_index.add_argument(
        "--strict", action="store_true", help="exit non-zero if any mismatch is found"
    )
    p_index.set_defaults(func=cmd_index)

    p_thumbs = sub.add_parser("thumbs", help="build or refresh the thumbnail cache")
    add_config(p_thumbs)
    p_thumbs.add_argument(
        "--workers",
        type=int,
        default=None,
        help="render threads (default: scales with CPU count)",
    )
    p_thumbs.add_argument("--force", action="store_true", help="re-render everything")
    p_thumbs.add_argument(
        "--no-autoscale",
        action="store_true",
        help="skip rendering the per-image autoscaled thumbnail variant",
    )
    p_thumbs.set_defaults(func=cmd_thumbs)

    p_gui = sub.add_parser("gui", help="open the browser")
    add_config(p_gui)
    p_gui.set_defaults(func=cmd_gui)

    p_check = sub.add_parser("check", help="print the last validation report")
    add_config(p_check)
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
