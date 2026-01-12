# Local Learning Coefficient (LLC) Estimation via SGLD

This branch implements an approach to measuring the complexity and "flatness" of the loss landscape in Transformer models.

By running **Stochastic Gradient Langevin Dynamics (SGLD)** chains initialized at pretrained checkpoints, we estimate the **Local Learning Coefficient (LLC)** ($\hat{\lambda}$). This metric serves as a proxy for the effective dimensionality of the local basin of attraction, offering insights into the developmental stages of the model.

## Core Methodology

The LLC is estimated using the formulation from Singular Learning Theory (SLT). We treat the optimization trajectory as a sampling process from a Gibbs distribution.

### 1. The Algorithm (SGLD)
We use a custom SGLD optimizer (`sgld.py`) that introduces a localization term (elasticity) to ensure the sampling chain stays within the local basin of the pretrained weights $w^*$:

$$\Delta w = - \frac{\epsilon}{2} \left( \nabla L(w) + \gamma (w - w^*) \right) + \eta$$

Where:
- $\nabla L(w)$ is the gradient of the loss.
- $\gamma$ (`elasticity`) is the spring constant anchoring the chain to $w^*$.
- $\eta \sim \mathcal{N}(0, \epsilon)$ is the injected Gaussian noise.

### 2. LLC Estimation
The `LLCEstimator` computes the complexity metric based on the difference between the average loss of the SGLD chain ($E[L]$) and the initial loss ($L(w^*)$):

$$\hat{\lambda} = n \beta (E[L] - L(w^*))$$

Where $n$ is the number of samples and $\beta$ is the inverse temperature.

## Installation

```
pip install torch transformers datasets pandas numpy tqdm huggingface_hub
```

## Usage
The main entry point is the CLI script (e.g., run_llc_cli.py). You can compute the LLC for the entire model or target specific layers and components to see how complexity varies across the architecture.

### Configuration
Hyperparameters for the SGLD chain are defined in the CONFIG dictionary within `hf_llc.py`:

```
CONFIG = {
    "lr": 1e-5,              # Step size (epsilon)
    "elasticity": 100.0,     # Localization strength (gamma)
    "temperature": "auto",   # Derived from batch size (log(BS))
    "num_samples": 25600,    # Total data points seen
    "num_steps": 400,        # Length of SGLD chain
    "burnin": 100            # Steps to discard before averaging
}
```

Output Results are saved as CSV files in the llc-bs64-steps400/ directory.
- Filename format: `llc_{component}.csv`
- Columns: revision, step, llc (the estimated metric), loss (initial loss).


## References & Acknowledgements

This implementation is based on the [devinterp](https://github.com/timaeus-research/devinterp) library by Timaeus Research.

**Primary Citations:**
* **Methodology:** Lau, E., et al. (2023). *The Local Learning Coefficient: A Singularity-Aware Complexity Measure for Deep Learning*. [arXiv:2308.12108](https://arxiv.org/abs/2308.12108)
* **SGLD & Development:** Hoogland, J., et al. (2024). *The Developmental Landscape of Few-Shot Learning*. [arXiv:2402.17937](https://arxiv.org/abs/2402.17937)

**Code Credit:**
The SGLD optimizer and LLC estimation logic in this repository are adapted from the [timaeus-research/devinterp](https://github.com/timaeus-research/devinterp) library.
