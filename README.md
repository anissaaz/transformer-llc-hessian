# Transformer Hessian & Loss Landscape Analysis

This repository contains tools for analyzing the geometry of the loss landscape in Transformer language models (specifically Pythia). It leverages [Curvlinops](https://github.com/f-dangel/curvlinops) to compute Hessian information, including eigenvalues, trace, and stable rank, as well as visualizing Hessian structure and clustering parameter trajectories during training.

## Features
- **Hessian Metric Estimation:** Computes top-k eigenvalues, Hutchinson trace estimation, and stable rank.
- **Visualizations:** Generates log-magnitude heatmaps of the Hessian matrix for specific layers or components.
- **Trajectory Clustering:** Clusters parameter trajectories using K-Means to identify developmental stages in model training.
- **Sub-matrix Analysis:** Supports targeted analysis of specific components (e.g., Attention Q/K/V, MLP expansion/contraction) rather than the full parameter set.

## Installation

Ensure you have a Python environment set up with PyTorch and the required dependencies:

```
pip install torch transformers datasets scipy pandas scikit-learn matplotlib huggingface_hub curvlinops 
```

## Running Experiments

The main entry point for running experiments is `run_transformer_cli.py`.  
This script allows you to specify which model components to analyze via command-line arguments.

---

## Quick Start

### Analyze the MLP Expansion Layer (Layer 5)

```
python run_transformer_cli.py \
  --model "EleutherAI/pythia-14m" \
  --layer 5 \
  --components mlp.exp \
  --size 1000
```

### Compare Query and Key Matrices  
Layer 0, Head 2:

```
python run_transformer_cli.py \
  -l 0 \
  -H 2 \
  -c attention.q attention.k \
  -s 500
```

## CLI Arguments

| Argument | Description | Default |
|--------|-------------|---------|
| `-m`, `--model` | Hugging Face model ID (e.g. `EleutherAI/pythia-70m-deduped`) | `EleutherAI/pythia-14m` |
| `-l`, `--layer` | Layer index to analyze (0–11 for Pythia-14m) | Required for Attn / MLP |
| `-H`, `--head` | Specific attention head index. If omitted, samples across all heads | All heads |
| `-s`, `--size` | Target size of the Hessian sub-matrix (number of parameters sampled) | 500 |
| `-c`, `--components` | List of components to include in the analysis (see below) | Required |

---

## Available Components

You can mix and match components using the `-c` flag:

- `embed_in`
- `embed_out`
- `attention.qkv` — Query, Key, Value together
- `attention.q` — Query matrix
- `attention.k` — Key matrix
- `attention.v` — Value matrix
- `attention.out` — Output projection
- `mlp.exp` — MLP expansion (`dense_h_to_4h`)
- `mlp.cont` — MLP contraction (`dense_4h_to_h`)

---

## Advanced Configuration

The core logic resides in `lm_hf_curvlinops.py`.  
You can control the experiment pipeline by modifying the global flags at the top of the file:


### Configuration Switches in lm_hf_curvlinops.py
```
COMPUTE_GRADS = False            # Save raw gradients for each step
COMPUTE_HESSIAN = True           # Required for metrics, plotting, or clustering
COMPUTE_HESSIAN_METRICS = False  # Calculate trace, max eigenvalue, stable rank
PLOT_HESSIAN = True              # Generate heatmaps (saved as .png)
CLUSTER_HESSIAN = False          # Run K-Means on diagonal trajectories
```