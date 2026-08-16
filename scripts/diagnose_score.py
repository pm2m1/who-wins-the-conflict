#!/usr/bin/env python
"""Standalone entry point equivalent to `python -m conflict_eval diagnose-score`.

Infrastructure validation only — scores two explicit candidate answers to
one question and prints a token-level breakdown. Not an experiment; output
is never written to results/.
"""

import argparse

from conflict_eval.cli import cmd_diagnose_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-b", required=True)
    args = parser.parse_args()
    cmd_diagnose_score(args.model, args.config, args.question, args.candidate_a, args.candidate_b)


if __name__ == "__main__":
    main()
