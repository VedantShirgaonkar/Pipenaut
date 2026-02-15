"""
Naive (Stop-and-Wait) pipeline schedule.

The simplest approach: forward the entire batch through all stages
sequentially, then backward through all stages sequentially.
Most stages are idle most of the time → ~75% bubble with 4 stages.
"""

import torch

from pipenaut.comms import PipelineComms
from pipenaut.model import ShardedMLP


def naive_pipeline_step(
    model: ShardedMLP,
    comms: PipelineComms,
    batch,
    targets,
    hidden_dim: int,
    chunks: int,  # unused — signature consistency with other schedules
    device: torch.device,
    profiler=None,
):
    """
    One training step using the Naive schedule.

    Flow:
      Forward:  Rank 0 → Rank 1 → ... → Rank N (sequential)
      Backward: Rank N → Rank N-1 → ... → Rank 0 (sequential)
    """
    # === FORWARD PASS ===

    if comms.rank == 0:
        input_data = batch
    else:
        shape = (batch, hidden_dim)
        if profiler:
            profiler.start("recv_fwd")
        input_data = comms.recv_forward(shape, device)
        if profiler:
            profiler.stop("recv_fwd")
        input_data.requires_grad = True

    if profiler:
        profiler.start("compute_fwd")
    output = model(input_data, targets)
    if profiler:
        profiler.stop("compute_fwd")

    if not model.is_last:
        if profiler:
            profiler.start("send_fwd")
        comms.send_forward(output.detach())
        if profiler:
            profiler.stop("send_fwd")

    # === BACKWARD PASS ===

    if model.is_last:
        loss = output
        if profiler:
            profiler.start("compute_bwd")
        loss.backward()
        if profiler:
            profiler.stop("compute_bwd")
    else:
        if profiler:
            profiler.start("recv_bwd")
        grad_from_next = comms.recv_backward(output.shape, device)
        if profiler:
            profiler.stop("recv_bwd")

        if profiler:
            profiler.start("compute_bwd")
        output.backward(grad_from_next)
        if profiler:
            profiler.stop("compute_bwd")

    if not model.is_first:
        if profiler:
            profiler.start("send_bwd")
        comms.send_backward(input_data.grad)
        if profiler:
            profiler.stop("send_bwd")

    if model.is_last:
        return loss
    return None
