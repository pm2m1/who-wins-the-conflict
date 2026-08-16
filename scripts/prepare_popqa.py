#!/usr/bin/env python
"""Standalone entry point equivalent to `python -m conflict_eval prepare-data`.

Thin wrapper: the actual logic lives in conflict_eval.cli.cmd_prepare_data,
so the CLI and this script cannot drift out of sync (docs/decisions.md).
"""

import argparse

from conflict_eval.cli import cmd_prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cmd_prepare_data(args.config)


if __name__ == "__main__":
    main()
