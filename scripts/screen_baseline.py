#!/usr/bin/env python
"""Standalone entry point equivalent to `python -m conflict_eval screen`."""

import argparse

from conflict_eval.cli import cmd_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cmd_screen(args.model, args.config)


if __name__ == "__main__":
    main()
