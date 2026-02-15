# pipenaut

**Experience pipeline parallelism on your laptop.** Naive, GPipe, 1F1B — one command.

No GPUs needed. No cluster setup. Just `pip install` and run.

## Quick Start

```bash
pip install pipenaut

# Compare all 3 pipeline schedules
pipenaut compare --workers 4

# Run a single schedule
pipenaut run --schedule 1f1b --workers 4 --steps 20

# Learn about a schedule
pipenaut explain 1f1b
```

## What is this?

Pipeline parallelism splits a neural network across multiple devices (or processes), with each device running one "stage" of the model. Data flows through the pipeline like an assembly line.

**pipenaut** lets you experience 3 classic pipeline schedules on your local CPU:

| Schedule | Strategy | Bubble Overhead |
|----------|----------|-----------------|
| **Naive** | Forward all → Backward all (sequential) | ~75% |
| **GPipe** | Forward all chunks → Backward all chunks | ~43% |
| **1F1B** | Interleave forward and backward | ~19% |

## Commands

### `pipenaut compare`

Runs all 3 schedules and prints a side-by-side comparison with pipeline timelines and a results table.

### `pipenaut run`

Run a single schedule with detailed output.

| Flag | Default | Description |
|------|---------|-------------|
| `--schedule, -s` | `1f1b` | Schedule: `naive`, `gpipe`, `1f1b` |
| `--workers, -w` | `4` | Number of pipeline stages |
| `--steps` | `30` | Training steps |
| `--chunks, -c` | `8` | Micro-batches |
| `--batch-size` | `32` | Batch size |
| `--dim` | `128` | Hidden dimension |
| `--layers` | `16` | Total model layers |

### `pipenaut explain`

Learn how a schedule works with ASCII diagrams and explanations.

```bash
pipenaut explain naive
pipenaut explain gpipe
pipenaut explain 1f1b
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Works on macOS, Linux, and Windows
- CPU only (CUDA used automatically if available)

## License

MIT
