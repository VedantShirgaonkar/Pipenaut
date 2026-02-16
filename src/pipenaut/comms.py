"""
Distributed communication primitives for pipeline parallelism.

Uses multiprocessing Queues for point-to-point data flow between
adjacent pipeline stages.  This avoids ``torch.distributed`` / Gloo
entirely, side-stepping hostname-resolution issues on Windows machines
with Docker Desktop, Hyper-V, or other virtual network adapters.
"""

import torch
import torch.multiprocessing as mp


class _DummyWork:
    """No-op handle returned by non-blocking sends for API compatibility."""

    def wait(self):
        pass


class PipelineComms:
    """
    Point-to-point communication handler for pipeline stages.

    Each stage only talks to its immediate neighbors:
      - send_forward / recv_forward: activations flow left → right
      - send_backward / recv_backward: gradients flow right → left

    Communication is backed by ``multiprocessing.Queue`` objects that
    are created by the runner and shared across workers.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        fwd_send_q=None,
        fwd_recv_q=None,
        bwd_send_q=None,
        bwd_recv_q=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < world_size - 1 else None
        self._fwd_send = fwd_send_q
        self._fwd_recv = fwd_recv_q
        self._bwd_send = bwd_send_q
        self._bwd_recv = bwd_recv_q

    def send_forward(self, tensor: torch.Tensor) -> None:
        """Send activations to the next stage (blocking)."""
        self._fwd_send.put(tensor.clone())

    def recv_forward(self, shape: tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        """Receive activations from the previous stage (blocking)."""
        return self._fwd_recv.get().to(device)

    def send_backward(self, tensor: torch.Tensor) -> None:
        """Send gradients to the previous stage (blocking)."""
        self._bwd_send.put(tensor.clone())

    def recv_backward(self, shape: tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        """Receive gradients from the next stage (blocking)."""
        return self._bwd_recv.get().to(device)

    def isend_forward(self, tensor: torch.Tensor):
        """Send activations to the next stage (non-blocking). Returns a no-op handle."""
        self._fwd_send.put(tensor.clone())
        return _DummyWork()
