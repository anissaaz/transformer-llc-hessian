# Transformer Developmental Landscape Analysis

This repository contains empirical research on the loss landscape geometry of Transformer language models (specifically Pythia). The project investigates **developmental interpretability** by analyzing how local curvature and complexity evolve during training.

The research is organized into two distinct branches, each focusing on a different methodological approach:

## 1. Hessian & Curvature Analysis
**Branch:** [`pythia-hessian-metrics`](https://github.com/anissaaz/transformer-llc-hessian/tree/pythia-hessian-metrics)

This branch contains the implementation for exact and approximate **Hessian analysis**.
* **Key Techniques:** Hutchinson's trace estimation, top-k eigenvalue computation, and parameter trajectory clustering (K-Means).
* **Goal:** To map the sharpening/flattening of specific model components (Attention vs. MLP) over time.

## 2. Local Learning Coefficient (LLC)
**Branch:** [`pythia-llc`](https://github.com/anissaaz/transformer-llc-hessian/tree/pythia-llc)

This branch implements **Stochastic Gradient Langevin Dynamics (SGLD)** to estimate the Local Learning Coefficient ($\hat{\lambda}$).
* **Key Techniques:** SGLD with elasticity (localization) and thermodynamic complexity estimation.
* **Goal:** To measure the effective dimensionality of the local basin of attraction as a proxy for model complexity.

---

### Citation & References
This work builds on research from the Singular Learning Theory community.
* **LLC Estimator:** Adapted from *Lau et al. (2023)* and the [DevInterp](https://github.com/timaeus-research/devinterp) library.
* **Curvature Analysis:** leverages [Curvlinops](https://github.com/f-dangel/curvlinops).