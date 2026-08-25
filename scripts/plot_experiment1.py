from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_GROUPS = {
    "sm_utilization": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput (% peak)"),
    "occupancy": ("sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved occupancy (%)"),
    "dram_throughput": ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput (% peak)"),
    "tensor_utilization": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "Tensor pipe active (%)"),
}


def read_ncu_metric(path: Path, metric_name: str) -> float | None:
    values = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(line for line in stream if not line.startswith("==")):
            if row.get("Metric Name") != metric_name:
                continue
            raw = (row.get("Metric Value") or "").replace(",", "").strip()
            try:
                values.append(float(raw))
            except ValueError:
                pass
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("profiling/figures/experiment1"))
    parser.add_argument(
        "--ncu-csv", action="append", default=[], metavar="BACKEND=PATH",
        help="Nsight Compute raw CSV; may be supplied once per backend",
    )
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    measured = [item for item in data["results"] if "median_us" in item]
    if not measured:
        raise SystemExit("result file does not contain benchmark measurements")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = [item["backend"] for item in measured]
    for field, ylabel, filename in (
        ("median_us", "Median latency (us)", "latency.png"),
        ("tflops", "TFLOPS", "tflops.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(names, [item[field] for item in measured])
        ax.set_ylabel(ylabel)
        ax.set_title("Experiment 1: irregular BF16 GEMMs")
        ax.bar_label(bars, fmt="%.2f")
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=180)
        plt.close(fig)

    ncu_files = {}
    for item in args.ncu_csv:
        backend, separator, path = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid --ncu-csv value: {item}")
        ncu_files[backend] = Path(path)

    for filename, (metric_name, ylabel) in METRIC_GROUPS.items():
        pairs = [
            (backend, read_ncu_metric(path, metric_name))
            for backend, path in ncu_files.items()
        ]
        pairs = [(backend, value) for backend, value in pairs if value is not None]
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        ax.set_ylabel(ylabel)
        ax.set_title("Experiment 1: H100 hardware profile")
        ax.bar_label(bars, fmt="%.2f")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{filename}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
