#!/usr/bin/env python3
"""Run both protein and genome scaling tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results")
    p.add_argument("--corpus", default="synthetic")
    p.add_argument("--max-sequences", type=int, default=5000)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    env = {"PYTHONPATH": str(root), **__import__("os").environ}

    for script, subdir in [
        ("run_tokenization_trap.py", "protein"),
        ("run_genome_bpe.py", "genome"),
    ]:
        cmd = [
            sys.executable,
            str(root / "experiments" / script),
            "--data-dir", args.data_dir,
            "--output", str(Path(args.output) / subdir),
            "--corpus", args.corpus,
            "--max-sequences", str(args.max_sequences),
        ]
        print(f"\n=== {subdir} ===")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
