from __future__ import annotations

import torch


@torch.no_grad()
def reference_outputs(problems) -> list[torch.Tensor]:
    return [(p.a.float() @ p.b.float()).to(torch.bfloat16) for p in problems]


@torch.no_grad()
def validate_outputs(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    atol: float = 0.5,
    rtol: float = 0.05,
) -> dict[str, object]:
    if len(actual) != len(expected):
        raise ValueError(f"output count mismatch: {len(actual)} != {len(expected)}")

    per_gemm = []
    all_correct = True
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        got_f = got.float()
        want_f = want.float()
        diff = (got_f - want_f).abs()
        relative = diff / want_f.abs().clamp_min(1.0e-6)
        finite = bool(torch.isfinite(got_f).all().item())
        correct = finite and torch.allclose(got_f, want_f, atol=atol, rtol=rtol)
        all_correct &= correct
        per_gemm.append({
            "gemm": index,
            "correct": correct,
            "finite": finite,
            "max_abs_error": float(diff.max().item()),
            "max_rel_error": float(relative.max().item()),
        })

    return {"correct": all_correct, "atol": atol, "rtol": rtol, "per_gemm": per_gemm}
