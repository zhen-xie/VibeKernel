from __future__ import annotations

from typing import Protocol

import torch


class Backend(Protocol):
    name: str

    def prepare(self, problems: list[object]) -> None: ...

    def run(self) -> list[torch.Tensor]: ...

    def metadata(self) -> dict[str, object]: ...

    def close(self) -> None: ...
