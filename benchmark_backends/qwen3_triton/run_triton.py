"""Pure-Triton timing entrypoint; reuses the PyTorch benchmark protocol."""
from pathlib import Path
import sys

REFERENCE = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
sys.path.insert(0, str(REFERENCE))
import run_pytorch
from triton_backend import TritonBackend

run_pytorch.PyTorchBackend = TritonBackend
run_pytorch.main()
