"""Entry point reserved for real-checkpoint Qwen3 MPK paged benchmarking.

The next implementation step binds this CLI to Mirage's registered Qwen3Builder
instead of duplicating its task graph in benchmark code.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import BenchmarkConfig

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Local Qwen3 checkpoint directory or Hugging Face model id")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args()
    cfg = BenchmarkConfig(**vars(args))
    print(f"MPK plan: model={cfg.model} B={cfg.batch} pages={cfg.max_num_pages} page_size={cfg.page_size}")
    print("Next: bind this configuration to the upstream Qwen3Builder and MPK runtime launcher.")

if __name__ == "__main__":
    main()
