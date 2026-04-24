"""Package entrypoint for `python -m impacts.emfac.activities`."""

from __future__ import annotations

import sys

from impacts.emfac.activities.main import main


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
