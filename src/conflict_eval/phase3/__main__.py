"""`python -m conflict_eval.phase3` entrypoint."""

from __future__ import annotations

import sys

from conflict_eval.phase3.cli import main

if __name__ == "__main__":
    sys.exit(main())
