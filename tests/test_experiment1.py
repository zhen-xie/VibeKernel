from types import SimpleNamespace

from vibekernel.experiments.experiment1.workload import description, total_flops


def test_total_flops() -> None:
    problems = [SimpleNamespace(m=4, n=16, k=8), SimpleNamespace(m=8, n=16, k=8)]
    assert total_flops(problems) == 2 * (4 + 8) * 16 * 8


def test_description() -> None:
    problems = [SimpleNamespace(m=4, n=4096, k=4096)]
    value = description(problems)
    assert value["m"] == [4]
    assert value["input_dtype"] == "bfloat16"
