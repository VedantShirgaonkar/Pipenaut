"""
Distributed communication primitives for pipeline parallelism.

Wraps torch.distributed point-to-point operations into a clean API
that handles forward (activations) and backward (gradients) data flow
between adjacent pipeline stages.
"""

import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initialize the distributed process group.

    Reads RANK, WORLD_SIZE, LOCAL_RANK from environment variables
    (set automatically by torch.multiprocessing.spawn or torchrun).

    Returns:
        tuple: (rank, world_size, device)
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Select the best available device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the process group with the appropriate backend
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, device


class PipelineComms:
    """
    Point-to-point communication handler for pipeline stages.

    Each stage only talks to its immediate neighbors:
      - send_forward / recv_forward: activations flow left → right
      - send_backward / recv_backward: gradients flow right → left
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor: torch.Tensor) -> None:
        """Send activations to the next stage (blocking)."""
        dist.send(tensor.contiguous(), dst=self.next_rank)

    def recv_forward(self, shape: tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        """Receive activations from the previous stage (blocking)."""
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor: torch.Tensor) -> None:
        """Send gradients to the previous stage (blocking)."""
        dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape: tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        """Receive gradients from the next stage (blocking)."""
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.next_rank)
        return tensor

    def isend_forward(self, tensor: torch.Tensor):
        """Send activations to the next stage (non-blocking). Returns request handle."""
        return dist.isend(tensor.contiguous(), dst=self.next_rank)
