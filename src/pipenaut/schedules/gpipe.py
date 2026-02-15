"""
GPipe pipeline schedule.

Splits the batch into micro-batches, forwards ALL chunks through the
pipeline, then backwards ALL chunks. Reduces bubble time compared to
Naive because stages overlap on different chunks.

Bubble overhead: (stages - 1) / (stages - 1 + chunks)
"""

import torch

from pipenaut.comms import PipelineComms
from pipenaut.model import ShardedMLP


def gpipe_pipeline_step(
    model: ShardedMLP,
    comms: PipelineComms,
    batch,
    targets,
    hidden_dim: int,
    chunks: int,
    device: torch.device,
    profiler=None,
):
    """
    One training step using the GPipe schedule.

    Flow:
      Phase 1 — Forward all chunks: fill the pipeline
      Phase 2 — Backward all chunks: drain the pipeline
    """
    # Prepare micro-batches
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)

    input_buffers = []
    output_buffers = []

    # === PHASE 1: ALL FORWARDS ===

    for i in range(chunks):
        if comms.rank == 0:
            input_data = micro_batches[i]
        else:
            shape = (batch // chunks, hidden_dim)
            if profiler:
                profiler.start("recv_fwd")
            input_data = comms.recv_forward(shape, device)
            if profiler:
                profiler.stop("recv_fwd")
            input_data.requires_grad = True

        if profiler:
            profiler.start("compute_fwd")
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[i])
        else:
            output = model(input_data)
        if profiler:
            profiler.stop("compute_fwd")

        if not model.is_last:
            if profiler:
                profiler.start("send_fwd")
            comms.send_forward(output.detach())
            if profiler:
                profiler.stop("send_fwd")

        input_buffers.append(input_data)
        output_buffers.append(output)

    # === PHASE 2: ALL BACKWARDS ===

    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)

    for i in range(chunks):
        input_data = input_buffers[i]
        output = output_buffers[i]

        if comms.rank == comms.world_size - 1:
            loss = output / chunks
            if profiler:
                profiler.start("compute_bwd")
            loss.backward()
            if profiler:
                profiler.stop("compute_bwd")
            total_loss += loss.detach()
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

        if comms.rank != 0:
            if profiler:
                profiler.start("send_bwd")
            comms.send_backward(input_data.grad)
            if profiler:
                profiler.stop("send_bwd")

    if comms.rank == comms.world_size - 1:
        return total_loss
    return None
