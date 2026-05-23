from __future__ import annotations

import sys

from impacts.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main(["emissions", *sys.argv[1:]]))
