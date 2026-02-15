# Pipenaut 

[![PyPI version](https://badge.fury.io/py/pipenaut.svg)](https://badge.fury.io/py/pipenaut)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> **Experience pipeline parallelism on your laptop.** Naive, GPipe, 1F1B — one command.

No GPUs needed. No cluster setup. just `pip install` and run.

---

##  Under the Hood

Visualizing how data flows through a pipeline is key to understanding efficiency.

![Pipenaut Workflow](docs/PP_pebble_graph.gif)

---

## Quick Start

```bash
pip install pipenaut

# Compare all 3 pipeline schedules side-by-side
pipenaut compare --workers 4

# Run a specific schedule with detailed logs
pipenaut run --schedule 1f1b --workers 4 --steps 20

# Learn how a schedule works
pipenaut explain 1f1b
```

---

## Supported Schedules

Pipenaut implements three classic pipeline parallelism strategies. You can visualize them directly in your terminal with `pipenaut explain <schedule>`.

### 1. Naive (Stop-and-Wait)
The simplest approach. Process one batch at a time through all stages. Massive idle time ("bubble").

```
[Rank 0] ████░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓
[Rank 1] ░░░░████░░░░░░░░░░░░░░░░▓▓▓▓░░░░
[Rank 2] ░░░░░░░░████░░░░░░░░▓▓▓▓░░░░░░░░
[Rank 3] ░░░░░░░░░░░░████▓▓▓▓░░░░░░░░░░░░

██ = Forward  ▓▓ = Backward  ░░ = Bubble (idle)
```

### 2. GPipe (Micro-batched)
Splits the batch into smaller "micro-batches". Pushes all micro-batches forward, then all backward. Much better utilization.

```
[Rank 0] ████████░░░░░░░░░░░░▓▓▓▓▓▓▓▓
[Rank 1] ░░████████░░░░░░░░▓▓▓▓▓▓▓▓░░
[Rank 2] ░░░░████████░░░░▓▓▓▓▓▓▓▓░░░░
[Rank 3] ░░░░░░████████▓▓▓▓▓▓▓▓░░░░░░
```

### 3. 1F1B (One Forward, One Backward)
The industry standard (used in Megatron-LM, DeepSpeed). Interleaves forward and backward passes to keep the pipeline full and memory usage low.

```
[Rank 0] ████████▓▓██▓▓██▓▓██▓▓██▓▓▓▓▓▓▓▓
[Rank 1] ░░██████▓▓██▓▓██▓▓██▓▓██▓▓██▓▓▓▓▓▓░░
[Rank 2] ░░░░████▓▓██▓▓██▓▓██▓▓██▓▓██▓▓██▓▓▓▓░░░░
[Rank 3] ░░░░░░██▓▓██▓▓██▓▓██▓▓██▓▓██▓▓██▓▓██▓▓░░░░░░
```

---

##  Comparison

| Schedule | Strategy | Bubble Overhead |
|----------|----------|-----------------|
| **Naive** | Forward all → Backward all (sequential) | ~75% |
| **GPipe** | Forward all chunks → Backward all chunks | ~43% |
| **1F1B** | Interleave forward and backward | ~19% |

---

## 🛠 Usage Reference

### `pipenaut run` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--schedule, -s` | `1f1b` | Schedule: `naive`, `gpipe`, `1f1b` |
| `--workers, -w` | `4` | Number of pipeline stages |
| `--steps` | `30` | Training steps |
| `--chunks, -c` | `8` | Micro-batches (for GPipe/1F1B) |
| `--batch-size` | `32` | Global batch size |
| `--dim` | `128` | Model hidden dimension |
| `--layers` | `16` | Total model layers |

---

## 📄 License

MIT © [Vedant Shirgaonkar](https://github.com/VedantShirgaonkar)
