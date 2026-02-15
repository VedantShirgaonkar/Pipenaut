"""
Rich terminal display for pipenaut.

Renders beautiful pipeline timelines, comparison tables,
progress indicators, and schedule explanations using the `rich` library.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(width=max(80, Console().width))

# ─── Color Palette ───────────────────────────────────────────────────────────

COLORS = {
    "forward": "bright_cyan",
    "backward": "bright_magenta",
    "bubble": "bright_black",
    "header": "bold bright_white",
    "accent": "bold bright_yellow",
    "success": "bold bright_green",
    "dim": "dim white",
    "loss": "bright_red",
}

SCHEDULE_DISPLAY_NAMES = {
    "naive": "Naive (Stop-and-Wait)",
    "gpipe": "GPipe (Micro-batched)",
    "1f1b": "1F1B (One Forward, One Backward)",
}

SCHEDULE_SHORT_NAMES = {
    "naive": "Naive",
    "gpipe": "GPipe",
    "1f1b": "1F1B",
}


# ─── Header ──────────────────────────────────────────────────────────────────

def print_header(workers: int, hidden_dim: int, total_layers: int, device_str: str = "CPU"):
    """Print the pipenaut header banner."""
    console.print()
    console.print(
        f"  [bold bright_cyan]🔧 pipenaut[/] [dim]— Pipeline Parallelism on Your Laptop[/]"
    )
    console.print(
        f"  [dim]   Model: {total_layers}-layer MLP ({hidden_dim}-dim) │ "
        f"{workers} pipeline stages │ {device_str}[/]"
    )
    console.print()


# ─── Pipeline Timeline ───────────────────────────────────────────────────────

def _build_naive_timeline(workers: int) -> list[str]:
    """Build ASCII timeline for Naive schedule."""
    total_slots = workers * 4  # each stage gets 2 fwd + 2 bwd slots, rest is bubble
    lines = []
    for rank in range(workers):
        bar = []
        for slot in range(total_slots):
            fwd_start = rank * 2
            fwd_end = fwd_start + 2
            bwd_start = (workers * 2) + (workers - 1 - rank) * 2
            bwd_end = bwd_start + 2

            if fwd_start <= slot < fwd_end:
                bar.append(("██", COLORS["forward"]))
            elif bwd_start <= slot < bwd_end:
                bar.append(("▓▓", COLORS["backward"]))
            else:
                bar.append(("░░", COLORS["bubble"]))
        lines.append((rank, bar))
    return lines


def _build_gpipe_timeline(workers: int, chunks: int) -> list[str]:
    """Build ASCII timeline for GPipe schedule."""
    display_chunks = min(chunks, 4)  # Cap visual chunks for readability
    warmup_slots = workers - 1
    fwd_slots = display_chunks
    bwd_slots = display_chunks
    total_slots = warmup_slots + fwd_slots + bwd_slots + warmup_slots

    lines = []
    for rank in range(workers):
        bar = []
        for slot in range(total_slots):
            # Forward phase: each rank starts 1 slot after the previous
            fwd_start = rank
            fwd_end = fwd_start + display_chunks

            # Backward phase: starts after all forwards, reverse order
            bwd_base = warmup_slots + fwd_slots
            bwd_start = bwd_base + (workers - 1 - rank)
            bwd_end = bwd_start + display_chunks

            if fwd_start <= slot < fwd_end:
                bar.append(("██", COLORS["forward"]))
            elif bwd_start <= slot < bwd_end:
                bar.append(("▓▓", COLORS["backward"]))
            else:
                bar.append(("░░", COLORS["bubble"]))
        lines.append((rank, bar))
    return lines


def _build_1f1b_timeline(workers: int, chunks: int) -> list[str]:
    """Build ASCII timeline for 1F1B schedule."""
    display_chunks = min(chunks, 8)
    lines = []
    for rank in range(workers):
        num_warmup = workers - rank - 1
        num_1f1b = display_chunks - num_warmup

        bar = []
        # Warmup bubble
        for _ in range(rank):
            bar.append(("░░", COLORS["bubble"]))
        # Warmup forwards
        for _ in range(num_warmup):
            bar.append(("██", COLORS["forward"]))
        # Steady state: 1F1B
        for _ in range(max(0, num_1f1b)):
            bar.append(("██", COLORS["forward"]))
            bar.append(("▓▓", COLORS["backward"]))
        # Cooldown backwards
        for _ in range(num_warmup):
            bar.append(("▓▓", COLORS["backward"]))
        # Trailing bubble
        remaining = rank
        for _ in range(remaining):
            bar.append(("░░", COLORS["bubble"]))

        lines.append((rank, bar))
    return lines


def print_timeline(schedule_name: str, workers: int, chunks: int = 8):
    """Print an ASCII pipeline timeline for a given schedule."""
    builders = {
        "naive": lambda: _build_naive_timeline(workers),
        "gpipe": lambda: _build_gpipe_timeline(workers, chunks),
        "1f1b": lambda: _build_1f1b_timeline(workers, chunks),
    }

    lines = builders[schedule_name]()

    display_name = SCHEDULE_DISPLAY_NAMES[schedule_name]
    console.print(f"  [dim]── {display_name} ──────────────────────────────────────[/]")

    for rank, bar in lines:
        rank_label = f"  [bold white]\\[Rank {rank}][/] "
        segments = Text()
        for char, color in bar:
            segments.append(char, style=color)
        console.print(rank_label, segments, sep="")

    console.print()
    console.print(
        f"  [{COLORS['forward']}]██[/] = Forward  "
        f"[{COLORS['backward']}]▓▓[/] = Backward  "
        f"[{COLORS['bubble']}]░░[/] = Bubble (idle)"
    )
    console.print()


# ─── Run Output ──────────────────────────────────────────────────────────────

def print_schedule_header(schedule_name: str):
    """Print a schedule section header."""
    display_name = SCHEDULE_DISPLAY_NAMES[schedule_name]
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print(f"  [bold bright_white] Schedule: {display_name}[/]")
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print()


def print_training_progress(losses: list[float], steps: int):
    """Print training loss values."""
    if not losses:
        return
    interval = max(1, len(losses) // 5)
    for i, loss_val in enumerate(losses):
        if i % interval == 0 or i == len(losses) - 1:
            console.print(f"  [dim]Step {i:03d}[/] │ Loss: [{COLORS['loss']}]{loss_val:.6f}[/]")


def print_run_stats(result: dict):
    """Print stats for a single run."""
    console.print()
    wall_time = result.get("wall_time", 0)
    bubble_pct = result.get("bubble_pct", 0)
    final_loss = result.get("final_loss", float("nan"))

    console.print(
        f"  ⏱  Time: [bold]{wall_time:.2f}s[/] │ "
        f"Bubble: [{COLORS['loss']}]{bubble_pct:.1f}%[/] │ "
        f"Final Loss: [{COLORS['loss']}]{final_loss:.6f}[/]"
    )
    console.print()


# ─── Comparison Table ────────────────────────────────────────────────────────

def print_comparison_progress(schedule_name: str, status: str = "running"):
    """Print progress line during comparison mode."""
    display_name = SCHEDULE_DISPLAY_NAMES[schedule_name]
    if status == "done":
        console.print(f"  [bright_green]✓[/] {display_name} [dim]... done[/]")
    else:
        console.print(f"  [bright_yellow]⏳[/] Running {display_name}...")


def print_comparison_table(results: list[dict]):
    """Print a beautiful comparison table of all schedule results."""
    console.print()
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print(f"  [bold bright_white] 📊 Results[/]")
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print()

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        padding=(0, 2),
        pad_edge=True,
    )

    table.add_column("Schedule", style="bold bright_cyan", no_wrap=True)
    table.add_column("Time", justify="right", no_wrap=True)
    table.add_column("Bubble", justify="right", no_wrap=True)
    table.add_column("Speedup", justify="right", no_wrap=True)
    table.add_column("Final Loss", justify="right", no_wrap=True)

    baseline_time = results[0].get("wall_time", 1) if results else 1

    for r in results:
        name = SCHEDULE_SHORT_NAMES.get(r["schedule"], r["schedule"])
        wall_time = r.get("wall_time", 0)
        bubble_pct = r.get("bubble_pct", 0)
        final_loss = r.get("final_loss", float("nan"))
        speedup = baseline_time / wall_time if wall_time > 0 else 0

        # Color the speedup based on how good it is
        if speedup >= 2.0:
            speedup_style = "bold bright_green"
        elif speedup > 1.0:
            speedup_style = "bright_yellow"
        else:
            speedup_style = "dim"

        table.add_row(
            name,
            f"{wall_time:.2f}s",
            f"{bubble_pct:.1f}%",
            f"[{speedup_style}]{speedup:.2f}x[/]",
            f"{final_loss:.6f}",
        )

    console.print(table)
    console.print()

    # Insight line — find the fastest schedule
    if len(results) >= 2:
        naive_time = results[0].get("wall_time", 1)
        best = min(results, key=lambda r: r.get("wall_time", float("inf")))
        best_time = best.get("wall_time", 1)
        speedup = naive_time / best_time if best_time > 0 else 1
        best_name = SCHEDULE_SHORT_NAMES.get(best["schedule"], best["schedule"])

        if speedup > 1.05:  # Only show if there's a meaningful difference
            console.print(
                f"  [bold bright_green]🏆[/] {best_name} is "
                f"[bold bright_green]{speedup:.2f}x faster[/] than Naive."
            )
        else:
            console.print(
                f"  [dim]💡 With this configuration, all schedules perform similarly.[/]"
            )
        console.print()


# ─── Explain ─────────────────────────────────────────────────────────────────

EXPLANATIONS = {
    "naive": {
        "title": "Naive Schedule (Stop-and-Wait)",
        "description": [
            "The simplest pipeline schedule. The entire batch flows through",
            "all stages one at a time:",
            "",
            "  1. Forward pass flows left → right through all stages",
            "  2. Only the LAST stage computes loss",
            "  3. Backward pass flows right → left through all stages",
            "  4. Most stages sit idle most of the time",
        ],
        "bubble_formula": "(stages - 1) / stages",
        "bubble_example": "75% with 4 stages",
        "used_by": "Nobody in practice — this is the baseline to understand\n"
                   "         why GPipe and 1F1B exist.",
    },
    "gpipe": {
        "title": "GPipe Schedule (Micro-batched)",
        "description": [
            "Splits the batch into smaller micro-batches and pipelines them:",
            "",
            "  1. Forward ALL micro-batches through the pipeline",
            "  2. Then backward ALL micro-batches",
            "  3. Stages overlap on different micro-batches → less idle time",
            "  4. But still has a gap between the forward and backward phases",
        ],
        "bubble_formula": "(stages - 1) / (stages - 1 + chunks)",
        "bubble_example": "~43% with 4 stages, 4 chunks",
        "used_by": "Google's GPipe paper (2019), some research systems.",
    },
    "1f1b": {
        "title": "1F1B Schedule (One Forward, One Backward)",
        "description": [
            "The most efficient basic schedule. After a warmup, each stage",
            "alternates one forward and one backward pass:",
            "",
            "  1. Warmup:       Fill the pipeline with forward passes",
            "  2. Steady state:  Alternate 1 forward + 1 backward per stage",
            "  3. Cooldown:      Drain remaining backward passes",
            "",
            "  Key insight: By starting backward passes before all forwards",
            "  are done, each stage stays busy during steady state.",
        ],
        "bubble_formula": "(stages - 1) / chunks",
        "bubble_example": "~19% with 4 stages, 8 chunks (using default pipenaut config)",
        "used_by": "Megatron-LM, DeepSpeed, most production pipeline systems.",
    },
}


def print_explain(schedule_name: str, workers: int = 4, chunks: int = 8):
    """Print a detailed explanation of a schedule with ASCII diagram."""
    if schedule_name not in EXPLANATIONS:
        console.print(f"  [bold red]Unknown schedule: {schedule_name}[/]")
        console.print(f"  [dim]Available: naive, gpipe, 1f1b[/]")
        return

    info = EXPLANATIONS[schedule_name]

    console.print()
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print(f"  [bold bright_white] 📖 {info['title']}[/]")
    console.print(f"  [bold bright_yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print()

    console.print(f"  [bold]How it works:[/]")
    for line in info["description"]:
        console.print(f"  [dim]{line}[/]" if line else "")
    console.print()

    # Show pipeline diagram
    console.print(f"  [bold]Pipeline diagram ({workers} stages):[/]")
    console.print()
    print_timeline(schedule_name, workers, chunks)

    console.print(f"  [bold]Bubble overhead:[/] [bright_red]{info['bubble_example']}[/]")
    console.print(f"  [bold]Formula:[/]         [dim]{info['bubble_formula']}[/]")
    console.print()
    console.print(f"  [bold]Used by:[/]  [dim]{info['used_by']}[/]")
    console.print()
