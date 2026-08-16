#!/usr/bin/env python
"""Standalone entry point equivalent to `python -m conflict_eval calibrate-sources`."""

import argparse

from conflict_eval.cli import cmd_calibrate_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cmd_calibrate_sources(args.model, args.config)


if __name__ == "__main__":
    main()
