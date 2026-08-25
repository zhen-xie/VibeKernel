from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from vibekernel.common.benchmark import measure_backend
from vibekernel.common.results import write_results
from vibekernel.common.validation import reference_outputs, validate_outputs
from vibekernel.experiments.experiment1.workload import create_workload, description, total_flops

BACKEND_NAMES = ("cublaslt_graph", "triton_grouped", "persistent_grouped")


def create_backend(name: str, args):
    if name == "cublaslt_graph":
        from .cublaslt_graph import CublasLtGraphBackend
        return CublasLtGraphBackend(stream_count=args.stream_count)
    if name == "triton_grouped":
        from .triton_grouped import TritonGroupedBackend
        return TritonGroupedBackend(
            block_m=args.triton_block_m,
            block_n=args.triton_block_n,
            block_k=args.triton_block_k,
            num_warps=args.triton_num_warps,
        )
    if name == "persistent_grouped":
        from .persistent_grouped import PersistentGroupedBackend
        return PersistentGroupedBackend(
            cutlass_path=args.cutlass_path,
            verbose_build=args.verbose_build,
        )
    raise ValueError(f"unknown backend: {name}")


def device_metadata() -> dict[str, object]:
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", *BACKEND_NAMES), default="all")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream-count", type=int, default=8)
    parser.add_argument("--cutlass-path", help="NVIDIA CUTLASS checkout; defaults to CUTLASS_PATH")
    parser.add_argument("--triton-block-m", type=int, default=16)
    parser.add_argument("--triton-block-n", type=int, default=64)
    parser.add_argument("--triton-block-k", type=int, default=32)
    parser.add_argument("--triton-num-warps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--profile", action="store_true", help="run a short NVTX-marked profiler workload")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    backends = BACKEND_NAMES if args.backend == "all" else (args.backend,)
    problems = create_workload(seed=args.seed)
    expected = reference_outputs(problems)
    results = []

    for name in backends:
        backend = create_backend(name, args)
        try:
            with torch.cuda.nvtx.range(f"experiment1/{name}/prepare"):
                backend.prepare(problems)
            with torch.cuda.nvtx.range(f"experiment1/{name}/validation"):
                actual = backend.run()
                torch.cuda.synchronize()
                correctness = validate_outputs(actual, expected, atol=args.atol, rtol=args.rtol)
            if not correctness["correct"]:
                raise RuntimeError(f"{name} failed correctness: {json.dumps(correctness)}")

            if args.profile:
                with torch.cuda.nvtx.range(f"experiment1/{name}/profile_region"):
                    for _ in range(10):
                        backend.run()
                    torch.cuda.synchronize()
                result = {"backend": name, "correctness": correctness, "metadata": backend.metadata()}
            else:
                with torch.cuda.nvtx.range(f"experiment1/{name}/measured"):
                    measurement = measure_backend(
                        backend, total_flops(problems), args.warmup,
                        args.iterations, args.repeats_per_sample,
                    )
                result = measurement.to_dict()
                result["correctness"] = correctness
                print(
                    f"{name:20s} median={measurement.median_us:10.3f} us "
                    f"mean={measurement.mean_us:10.3f} us TFLOPS={measurement.tflops:8.3f}"
                )
            results.append(result)
        finally:
            backend.close()

    payload = {
        "experiment": description(problems),
        "device": device_metadata(),
        "profile_mode": args.profile,
        "results": results,
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(f"results/experiment1/{stamp}.json")
    path = write_results(payload, output)
    print(f"results: {path}")


if __name__ == "__main__":
    main()
