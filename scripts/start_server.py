#!/usr/bin/env python3
"""Launch the GR00T N1.7 policy server."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zerohlm_bench.launch import main

if __name__ == "__main__":
    raise SystemExit(main("server", ROOT))
