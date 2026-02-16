"""
Process runner for pipeline parallelism.

Spawns multiple processes using torch.multiprocessing to simulate
pipeline stages. Inter-process communication uses multiprocessing
Queues — no torch.distributed / Gloo networking is involved,
so this works on every OS regardless of Docker, Hyper-V, or
other virtual network adapter configurations.
"""

import time

import torch
import torch.multiprocessing as mp
import torch.optim as optim

from pipenaut.comms import PipelineComms
from pipenaut.model import ShardedMLP
from pipenaut.profiler import PipelineProfiler
from pipenaut.schedules import SCHEDULES


def _worker(
    rank: int,
    world_size: int,
    schedule_name: str,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    total_layers: int,
    chunks: int,
    fwd_queues: list,
    bwd_queues: list,
    result_dict: dict,
):
    """
    Worker function that runs on each spawned process.

    Sets up the model shard, runs the training loop, and stores
    results for the main process.
    """
    device = torch.device("cpu")

    # Build per-rank queue pairs
    fwd_send_q = fwd_queues[rank] if rank < world_size - 1 else None
    fwd_recv_q = fwd_queues[rank - 1] if rank > 0 else None
    bwd_send_q = bwd_queues[rank - 1] if rank > 0 else None
    bwd_recv_q = bwd_queues[rank] if rank < world_size - 1 else None

    comms = PipelineComms(
        rank, world_size,
        fwd_send_q=fwd_send_q,
        fwd_recv_q=fwd_recv_q,
        bwd_send_q=bwd_send_q,
        bwd_recv_q=bwd_recv_q,
    )

    # Reproducible initialization
    torch.manual_seed(42 + rank)

    # Create model shard
    model = ShardedMLP(hidden_dim, total_layers, rank, world_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data (only on relevant ranks)
    if rank == 0:
        fixed_input = torch.randn(batch_size, hidden_dim, device=device)
    else:
        fixed_input = batch_size  # Pass batch_size as int for buffer allocation

    if rank == world_size - 1:
        fixed_target = torch.randint(0, 2, (batch_size,), device=device)
    else:
        fixed_target = None

    # Get the schedule function
    schedule_fn = SCHEDULES[schedule_name]

    # Profiler
    profiler = PipelineProfiler()

    # Training loop
    model.train()
    losses = []
    wall_start = time.perf_counter()

    for step in range(steps):
        optimizer.zero_grad()
        profiler.start_step()

        loss = schedule_fn(
            model=model,
            comms=comms,
            batch=fixed_input,
            targets=fixed_target,
            hidden_dim=hidden_dim,
            chunks=chunks,
            device=device,
            profiler=profiler,
        )

        profiler.stop_step()
        optimizer.step()

        if rank == world_size - 1 and loss is not None:
            losses.append(loss.item())

    wall_time = time.perf_counter() - wall_start

    # Store results (only last rank has loss, all ranks have timing)
    stats = profiler.get_stats()
    result_dict[f"rank_{rank}_stats"] = stats
    result_dict[f"rank_{rank}_wall_time"] = wall_time

    if rank == world_size - 1:
        result_dict["losses"] = losses
        result_dict["wall_time"] = wall_time
        result_dict["final_loss"] = losses[-1] if losses else float("nan")
        result_dict["bubble_pct"] = stats["bubble_pct"]


def run_schedule(
    schedule_name: str,
    workers: int = 4,
    steps: int = 30,
    batch_size: int = 32,
    hidden_dim: int = 128,
    total_layers: int = 16,
    chunks: int = 8,
) -> dict:
    """
    Run a pipeline schedule and return results.

    Spawns `workers` processes, each running one pipeline stage,
    trains for `steps` iterations, and returns timing/loss data.

    Args:
        schedule_name: One of "naive", "gpipe", "1f1b"
        workers: Number of pipeline stages
        steps: Training iterations
        batch_size: Batch size
        hidden_dim: Hidden layer dimension
        total_layers: Total layers in the MLP (split across stages)
        chunks: Number of micro-batches (for gpipe/1f1b)

    Returns:
        dict with keys: schedule, wall_time, final_loss, bubble_pct,
                        losses, per-rank stats
    """
    if schedule_name not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{schedule_name}'. Choose from: {list(SCHEDULES.keys())}")

    if total_layers % workers != 0:
        raise ValueError(f"total_layers ({total_layers}) must be divisible by workers ({workers})")

    if batch_size % chunks != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by chunks ({chunks})")

    # Create queues for inter-stage communication.
    # fwd_queues[i] carries activations from rank i → rank i+1
    # bwd_queues[i] carries gradients  from rank i+1 → rank i
    fwd_queues = [mp.Queue() for _ in range(workers - 1)]
    bwd_queues = [mp.Queue() for _ in range(workers - 1)]

    # Use a multiprocessing Manager to share results across processes
    manager = mp.Manager()
    result_dict = manager.dict()

    # Spawn workers
    mp.spawn(
        _worker,
        args=(
            workers,
            schedule_name,
            steps,
            batch_size,
            hidden_dim,
            total_layers,
            chunks,
            fwd_queues,
            bwd_queues,
            result_dict,
        ),
        nprocs=workers,
        join=True,
    )

    # Convert manager dict to regular dict
    results = dict(result_dict)
    results["schedule"] = schedule_name
    results["workers"] = workers
    results["steps"] = steps
    results["batch_size"] = batch_size
    results["hidden_dim"] = hidden_dim
    results["total_layers"] = total_layers
    results["chunks"] = chunks

    return results
