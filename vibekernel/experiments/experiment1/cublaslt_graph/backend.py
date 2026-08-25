from __future__ import annotations

import torch


class CublasLtGraphBackend:
    """Multi-stream PyTorch matmuls captured and replayed as one CUDA Graph.

    On CUDA builds PyTorch dispatches these BF16 matrix multiplies to its
    cuBLAS/cuBLASLt path. Nsight should be used to record the selected kernels.
    """

    name = "cublaslt_graph"

    def __init__(self, stream_count: int = 8) -> None:
        self.stream_count = stream_count
        self._outputs: list[torch.Tensor] = []
        self._streams: list[torch.cuda.Stream] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        self._replay_stream: torch.cuda.Stream | None = None
        self._problems = []

    def prepare(self, problems) -> None:
        if not problems:
            raise ValueError("workload is empty")
        if self.stream_count < 1:
            raise ValueError("stream_count must be positive")

        self._problems = list(problems)
        preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
        if preferred_blas is not None:
            preferred_blas("cublaslt")
        self._outputs = [torch.empty((p.m, p.n), device=p.a.device, dtype=torch.bfloat16) for p in problems]
        count = min(self.stream_count, len(problems))
        self._streams = [torch.cuda.Stream() for _ in range(count)]
        self._replay_stream = torch.cuda.Stream()

        # Initialize library handles, algorithms, and allocator state before capture.
        for p, out in zip(problems, self._outputs, strict=True):
            torch.mm(p.a, p.b, out=out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        replay_stream = self._replay_stream
        current = torch.cuda.current_stream()
        replay_stream.wait_stream(current)
        with torch.cuda.stream(replay_stream):
            graph.capture_begin()
            for index, (p, out) in enumerate(zip(problems, self._outputs, strict=True)):
                worker = self._streams[index % count]
                worker.wait_stream(replay_stream)
                with torch.cuda.stream(worker):
                    torch.mm(p.a, p.b, out=out)
            for worker in self._streams:
                replay_stream.wait_stream(worker)
            graph.capture_end()
        current.wait_stream(replay_stream)
        torch.cuda.synchronize()
        self._graph = graph

    def run(self) -> list[torch.Tensor]:
        if self._graph is None or self._replay_stream is None:
            raise RuntimeError("backend has not been prepared")
        # The graph is tied to its non-default capture stream. Fork from the
        # caller's timing stream and join back so surrounding CUDA Events cover
        # the entire replay rather than only the host submission.
        caller = torch.cuda.current_stream()
        self._replay_stream.wait_stream(caller)
        self._graph.replay()
        caller.wait_stream(self._replay_stream)
        return self._outputs

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "PyTorch CUDA matmul (cuBLAS/cuBLASLt)",
            "scheduling": "multi-stream CUDA Graph replay",
            "stream_count": len(self._streams),
            "graph_replay": True,
        }

    def close(self) -> None:
        self._graph = None
        self._streams.clear()
        self._outputs.clear()
        self._problems.clear()
