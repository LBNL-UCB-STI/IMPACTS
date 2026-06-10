from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.profile_runner",
        description="Internal profiling wrapper for python -m impacts commands.",
    )
    parser.add_argument("--pstats", required=True)
    parser.add_argument("command_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command_args = list(args.command_args)
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    if not command_args:
        parser.error("missing impacts command arguments after --")

    profile_path = Path(args.pstats).expanduser().resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    exit_code = 1
    profiler.enable()
    try:
        from impacts.__main__ import main as impacts_main

        exit_code = int(impacts_main(command_args))
    except SystemExit as exc:
        code = exc.code
        exit_code = int(code) if isinstance(code, int) else 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        profiler.disable()
        profiler.dump_stats(str(profile_path))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
