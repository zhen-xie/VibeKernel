from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class BenchmarkConfig:
    model: str
    batch: int = 1
    prompt_len: int = 512
    decode_steps: int = 32
    page_size: int = 64
    warmup: int = 20
    repeats: int = 100

    @property
    def max_seq_len(self) -> int:
        return self.prompt_len + self.decode_steps

    @property
    def pages_per_request(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size

    @property
    def max_num_pages(self) -> int:
        return self.batch * self.pages_per_request

    def write_json(self, path: Path, result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": asdict(self), **result}, indent=2) + "\n")
