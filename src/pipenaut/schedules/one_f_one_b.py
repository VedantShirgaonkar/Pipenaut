"""
1F1B (One Forward, One Backward) pipeline schedule.

The most efficient schedule: after a warmup phase, each stage
alternates one forward and one backward pass, keeping all stages
busy during steady state.

Bubble overhead: (stages - 1) / chunks
"""

import torch

from pipenaut.comms import PipelineComms
from pipenaut.model import ShardedMLP


def onef_oneb_pipeline_step(
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
    One training step using the 1F1B schedule.

    Flow:
      Phase 1 — Warmup:       forward-only to fill the pipeline
      Phase 2 — Steady state:  alternate 1 forward + 1 backward
      Phase 3 — Cooldown:      backward-only to drain the pipeline
    """
    # Prepare micro-batches
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)

    input_buffers = [None] * chunks
    output_buffers = [None] * chunks
    async_requests = []

    # Schedule parameters
    num_warmup = comms.world_size - comms.rank - 1
    num_1f1b = chunks - num_warmup

    def run_forward(idx):
        if comms.rank == 0:
            input_data = micro_batches[idx]
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
            output = model(input_data, micro_targets[idx])
        else:
            output = model(input_data)
        if profiler:
            profiler.stop("compute_fwd")

        if not model.is_last:
            if profiler:
                profiler.start("send_fwd")
            req = comms.isend_forward(output.detach())
            async_requests.append(req)
            if profiler:
                profiler.stop("send_fwd")

        input_buffers[idx] = input_data
        output_buffers[idx] = output

    def run_backward(idx):
        input_data = input_buffers[idx]
        output = output_buffers[idx]

        if comms.rank == comms.world_size - 1:
            loss = output / chunks
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

        if comms.rank != 0:
            if profiler:
                profiler.start("send_bwd")
            comms.send_backward(input_data.grad)
            if profiler:
                profiler.stop("send_bwd")

        if comms.rank == comms.world_size - 1:
            return loss
        return None

    # === EXECUTION ===

    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)

    # Phase 1: Warmup (forward only)
    for i in range(num_warmup):
        run_forward(i)

    # Phase 2: Steady state (1 forward + 1 backward)
    for i in range(num_1f1b):
        run_forward(i + num_warmup)
        res = run_backward(i)
        if comms.rank == comms.world_size - 1:
            total_loss += res.detach()

    # Phase 3: Cooldown (backward only)
    for i in range(num_warmup):
        res = run_backward(i + num_1f1b)
        if comms.rank == comms.world_size - 1:
            total_loss += res.detach()

    if comms.rank == comms.world_size - 1:
        return total_loss
    return None
