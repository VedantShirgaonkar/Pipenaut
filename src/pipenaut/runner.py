"""
Process runner for pipeline parallelism.

Spawns multiple processes using torch.multiprocessing to simulate
pipeline stages. The user never needs to touch torchrun — this
module handles all the distributed boilerplate.
"""

import os
import time

import torch
import torch.distributed as dist
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
    result_dict: dict,
):
    """
    Worker function that runs on each spawned process.

    Sets up the distributed environment, creates the model shard,
    runs the training loop, and stores results for the main process.
    """
    # Set environment variables for init_distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(result_dict.get("_port", 29500))

    # Select device
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        torch.cuda.set_device(device)

    comms = PipelineComms(rank, world_size)

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

    dist.destroy_process_group()


def _find_free_port():
    """Find a free port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


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

    # Use a multiprocessing Manager to share results across processes
    manager = mp.Manager()
    result_dict = manager.dict()
    result_dict["_port"] = _find_free_port()

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
