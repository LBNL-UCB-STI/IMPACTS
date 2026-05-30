"""Package entrypoint for `python -m impacts.pipeline.emfac.fleet`."""

from __future__ import annotations

import sys

from impacts.pipeline.emfac.fleet.main import main


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
