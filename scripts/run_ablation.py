#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--name", required=True,
                    choices=["no_graph", "no_contrastive", "standard_lora", "no_phonology"])
    args = ap.parse_args()
    cmd = [sys.executable, "scripts/train.py", "--config", args.config, "--ablation", args.name]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
