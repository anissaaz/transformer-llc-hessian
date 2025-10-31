import os
import pandas as pd

# Use a non-interactive backend before importing pyplot
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

csv_path = "pythia_hessian_metrics.csv"
assert os.path.exists(csv_path), f"Missing CSV at {csv_path}"
df = pd.read_csv(csv_path)

# Optional sanity print
print(df.head(3).to_string(index=False))

# Linear scale
plt.figure(figsize=(11,6))
plt.plot(df["step"], df["max_eig"], label="Max eigenvalue", lw=2)
plt.plot(df["step"], df["stable_rank"], label="Stable rank", lw=2)
plt.plot(df["step"], df["trace"], label="Trace", lw=2)
plt.xlabel("Training Step")
plt.ylabel("Metric Value")
plt.title("Curvature metrics over training (Pythia-70M)")
plt.legend()
plt.grid(True)
out1 = "pythia_hessian_metrics_linear.png"
plt.savefig(out1, dpi=220, bbox_inches="tight")
print(f"Saved {out1}")

# Log scale version (often useful when magnitudes differ)
plt.figure(figsize=(11,6))
plt.plot(df["step"], df["max_eig"], label="Max eigenvalue", lw=2)
plt.plot(df["step"], df["stable_rank"], label="Stable rank", lw=2)
plt.plot(df["step"], df["trace"], label="Trace", lw=2)
plt.xlabel("Training Step")
plt.ylabel("Metric Value (log scale)")
plt.title("Curvature metrics over training (Pythia-70M) – log scale")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
out2 = "pythia_hessian_metrics_log.png"
plt.savefig(out2, dpi=220, bbox_inches="tight")
print(f"Saved {out2}")