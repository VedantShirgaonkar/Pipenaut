"""
CLI entry point for pipenaut.

Provides three commands:
  pipenaut run      — Run a single pipeline schedule
  pipenaut compare  — Run all 3 schedules and compare
  pipenaut explain  — Learn about a schedule
"""

import argparse
import sys

import torch

from pipenaut import __version__
from pipenaut.display import (
    console,
    print_comparison_progress,
    print_comparison_table,
    print_explain,
    print_header,
    print_run_stats,
    print_schedule_header,
    print_timeline,
    print_training_progress,
)
from pipenaut.runner import run_schedule


def _detect_device() -> str:
    """Detect the best available device for display purposes."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        return f"CUDA ({name} x{count})" if count > 1 else f"CUDA ({name})"
    return "CPU"


def cmd_run(args):
    """Handle the 'run' subcommand."""
    device_str = _detect_device()
    print_header(args.workers, args.dim, args.layers, device_str)
    print_schedule_header(args.schedule)

    console.print(f"  [dim]Training for {args.steps} steps with {args.chunks} micro-batches...[/]")
    console.print()

    result = run_schedule(
        schedule_name=args.schedule,
        workers=args.workers,
        steps=args.steps,
        batch_size=args.batch_size,
        hidden_dim=args.dim,
        total_layers=args.layers,
        chunks=args.chunks,
    )

    # Show training progress
    losses = result.get("losses", [])
    print_training_progress(losses, args.steps)

    # Show pipeline timeline
    console.print()
    console.print(f"  [bold bright_white]📊 Pipeline Timeline[/]")
    console.print()
    print_timeline(args.schedule, args.workers, args.chunks)

    # Show stats
    print_run_stats(result)


def cmd_compare(args):
    """Handle the 'compare' subcommand."""
    device_str = _detect_device()
    print_header(args.workers, args.dim, args.layers, device_str)

    schedules = ["naive", "gpipe", "1f1b"]
    results = []

    console.print(f"  [dim]Running all 3 schedules ({args.steps} steps each)...[/]")
    console.print()

    for schedule_name in schedules:
        print_comparison_progress(schedule_name, status="running")

        result = run_schedule(
            schedule_name=schedule_name,
            workers=args.workers,
            steps=args.steps,
            batch_size=args.batch_size,
            hidden_dim=args.dim,
            total_layers=args.layers,
            chunks=args.chunks,
        )
        results.append(result)

        wall_time = result.get("wall_time", 0)
        print_comparison_progress(schedule_name, status="done")

    # Show timelines for all schedules
    console.print()
    console.print(f"  [bold bright_white]📊 Pipeline Timelines[/]")
    console.print()
    for schedule_name in schedules:
        print_timeline(schedule_name, args.workers, args.chunks)

    # Show comparison table
    print_comparison_table(results)


def cmd_explain(args):
    """Handle the 'explain' subcommand."""
    print_explain(args.schedule, workers=args.workers, chunks=args.chunks)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pipenaut",
        description="🔧 pipenaut — Experience pipeline parallelism on your laptop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pipenaut run --schedule 1f1b --workers 4
  pipenaut compare --workers 4 --steps 20
  pipenaut explain naive
  pipenaut explain 1f1b
        """,
    )
    parser.add_argument("--version", action="version", version=f"pipenaut {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── run ──────────────────────────────────────────────────────────────

    run_parser = subparsers.add_parser("run", help="Run a single pipeline schedule")
    run_parser.add_argument(
        "--schedule", "-s",
        choices=["naive", "gpipe", "1f1b"],
        default="1f1b",
        help="Pipeline schedule to use (default: 1f1b)",
    )
    run_parser.add_argument("--workers", "-w", type=int, default=4, help="Number of pipeline stages (default: 4)")
    run_parser.add_argument("--steps", type=int, default=30, help="Training steps (default: 30)")
    run_parser.add_argument("--chunks", "-c", type=int, default=8, help="Number of micro-batches (default: 8)")
    run_parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    run_parser.add_argument("--dim", type=int, default=128, help="Hidden dimension (default: 128)")
    run_parser.add_argument("--layers", type=int, default=16, help="Total model layers (default: 16)")

    # ─── compare ──────────────────────────────────────────────────────────

    compare_parser = subparsers.add_parser("compare", help="Compare all 3 schedules side-by-side")
    compare_parser.add_argument("--workers", "-w", type=int, default=4, help="Number of pipeline stages (default: 4)")
    compare_parser.add_argument("--steps", type=int, default=30, help="Training steps (default: 30)")
    compare_parser.add_argument("--chunks", "-c", type=int, default=8, help="Number of micro-batches (default: 8)")
    compare_parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    compare_parser.add_argument("--dim", type=int, default=128, help="Hidden dimension (default: 128)")
    compare_parser.add_argument("--layers", type=int, default=16, help="Total model layers (default: 16)")

    # ─── explain ──────────────────────────────────────────────────────────

    explain_parser = subparsers.add_parser("explain", help="Learn about a pipeline schedule")
    explain_parser.add_argument(
        "schedule",
        choices=["naive", "gpipe", "1f1b"],
        help="Schedule to explain",
    )
    explain_parser.add_argument("--workers", "-w", type=int, default=4, help="Workers for diagram (default: 4)")
    explain_parser.add_argument("--chunks", "-c", type=int, default=8, help="Chunks for diagram (default: 8)")

    # ─── Parse & dispatch ─────────────────────────────────────────────────

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Validate constraints
    if args.command in ("run", "compare"):
        if args.layers % args.workers != 0:
            console.print(
                f"[bold red]Error:[/] --layers ({args.layers}) must be divisible by --workers ({args.workers})"
            )
            sys.exit(1)
        if args.batch_size % args.chunks != 0:
            console.print(
                f"[bold red]Error:[/] --batch-size ({args.batch_size}) must be divisible by --chunks ({args.chunks})"
            )
            sys.exit(1)

    commands = {
        "run": cmd_run,
        "compare": cmd_compare,
        "explain": cmd_explain,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
