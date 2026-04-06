""" 
Loads the Adam v_t trajectory and produces two plots:
 
  1. Mean v_t across training steps — one line per block type (Q/K/V split)
  2. CV (std/mean) across training steps — stratified by block group
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ===========================================================================
# CONFIGURATION
# ===========================================================================
MODEL = "70m-seed1" # or 410m
TRAJECTORY_PATH = f"extracted_adam_states/adam_second_moment_trajectory_{MODEL}.pt"
OUTPUT_DIR      = Path("second_moment_figs")
N_LAYERS        = 6
LAYER_OFFSET    = 2 # adam key prefix for layer 0 is "2.xxx"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameter suffixes and how to label them
# Tuples of (adam_suffix, pretty_name, split_qkv)
# ---------------------------------------------------------------------------

PARAM_BLOCKS = [
    ("attention.query_key_value.weight", "QKV_W"      ),
    ("attention.query_key_value.bias",   "QKV_Bias"   ),
    ("attention.dense.weight",           "Attn_Out_W" ),
    ("attention.dense.bias",             "Attn_Out_B" ),
    ("mlp.dense_h_to_4h.weight",         "MLP_Exp_W"  ),
    ("mlp.dense_h_to_4h.bias",           "MLP_Exp_B"  ),
    ("mlp.dense_4h_to_h.weight",         "MLP_Cont_W" ),
    ("mlp.dense_4h_to_h.bias",           "MLP_Cont_B" ),
    ("input_layernorm.weight",           "LN1_W"      ),
    ("input_layernorm.bias",             "LN1_B"      ),
    ("post_attention_layernorm.weight",  "LN2_W"      ),
    ("post_attention_layernorm.bias",    "LN2_B"      ),
]

# ---------------------------------------------------------------------------
# Load trajectory
# ---------------------------------------------------------------------------
print("Loading trajectory...")
traj = torch.load(TRAJECTORY_PATH, map_location="cpu")
steps  = traj["steps"]
states = traj["states"]
print(f"  {len(steps)} steps: {steps[0]} → {steps[-1]}")

# ---------------------------------------------------------------------------
# Build a flat records list — one row per (step, layer, block_type, element)
# aggregated to (step, layer, block_type) with mean and std stored
# ---------------------------------------------------------------------------
records = []

for step in steps:
    state = states[step]
    
    for layer_idx in range(N_LAYERS):
        for suffix, pretty in PARAM_BLOCKS:
            
            key = f"{layer_idx + LAYER_OFFSET}.{suffix}"
            if key not in state:
                continue
 
            tensor = state[key].float()
            
            if pretty == "QKV_W":
                # QKV weight: shape [3H, H] — split along dim 0
                
                chunks = tensor.chunk(3, dim=0)
                sub_blocks = [("Q_W", chunks[0]), ("K_W", chunks[1]), ("V_W", chunks[2])]              
            else:
                sub_blocks = [(pretty, tensor)]
                
            for block_name, t in sub_blocks:
                v = t.flatten()
                v = v[v > 1e-12]
                if v.numel() < 10:
                    continue
            
                mean_val = v.mean().item()
                std_val = v.std().item()

                records.append({
                    "step":    step,
                    "layer":   layer_idx,
                    "block":   block_name, # This will be Q_W, K_W, V_W, or the original pretty_name
                    "mean_vt": mean_val,
                    "std_vt":  std_val,
                    "cv":      std_val / mean_val if mean_val > 0 else np.nan,  
                })

df = pd.DataFrame(records)
print(f"  Built DataFrame: {len(df)} rows")


# ---------------------------------------------------------------------------
# Plot 1 — Mean v_t across steps, Q/K/V + MLP weights only
# ---------------------------------------------------------------------------
WEIGHT_BLOCKS = ["Q_W", "K_W", "V_W", "Attn_Out_W", "MLP_Exp_W", "MLP_Cont_W"]
LAYER_TO_ANALYZE = 0

# Specify filters
df_specific = df[
    (df["layer"].isin([LAYER_TO_ANALYZE])) #& 
    # (df["block"].isin(["Q_W", "K_W", "V_W"]))
]

# only use with sns lineplot
df_weights = df_specific[df_specific["block"].isin(WEIGHT_BLOCKS)].copy()

# Average across layers so one line = one block type
df_weights_avg = (
    df_weights
    .groupby(["step", "block"])["mean_vt"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))

# --- mean lines –--
sns.lineplot(
    data=df_weights_avg,
    x="step", y="mean_vt",
    hue="block",
    #palette=WEIGHT_PALETTE,
    hue_order=WEIGHT_BLOCKS,
    linewidth=2,
    marker="o",
    markersize=3,
    ax=ax,
)

# --- mean lines with std –--
# for block in WEIGHT_BLOCKS:
#     df_block = df_specific[df_specific["block"] == block]
    
#     if df_block.empty: 
#         continue
        
#     line, = ax.plot(
#         df_block["step"], df_block["mean_vt"], 
#         label=block, linewidth=2
#     )
    
#     # Draw Shaded Band (Mean +/- 1 Standard Deviation); 1 std: 68% of tensor
#     ax.fill_between(
#         df_block["step"],
#         df_block["mean_vt"] - df_block["std_vt"], # Lower bound
#         df_block["mean_vt"] + df_block["std_vt"], # Upper bound
#         color=line.get_color(),                   # Match band color to line color
#         alpha=0.2                                 # 20% opacity so you can see through it
#     )
    
    
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Mean $v_t$ (log scale)")
ax.set_title(f"Adam second moment L{LAYER_TO_ANALYZE} — Pythia-{MODEL}")
#ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend(title="Block type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.tight_layout()
path = OUTPUT_DIR / f"mean_vt_{MODEL}_L{LAYER_TO_ANALYZE}.png"
#plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — CV across steps, stratified by group
# Mean CV across layers, shaded band = ±1 std across layers
# ---------------------------------------------------------------------------
WEIGHT_BLOCKS = ["Q_W", "K_W", "V_W", "Attn_Out_W", "MLP_Exp_W", "MLP_Cont_W"]
LAYER_TO_ANALYZE = 0

df_cv = df[
    (df["layer"] == LAYER_TO_ANALYZE) & 
    (df["block"].isin(WEIGHT_BLOCKS))
].dropna(subset=["cv"]).copy()


fig, ax = plt.subplots(figsize=(11, 5))

sns.lineplot(
    data=df_cv,
    x="step", 
    y="cv", 
    hue="block",
    linewidth=2,
    marker="o",
    markersize=3,
    ax=ax
)

ax.set_xscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("CV of $v_t$ (std / mean within block)")
ax.set_title(f"Adam $v_t$ heterogeneity across training L{LAYER_TO_ANALYZE} — Pythia-{MODEL}")

ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.legend(title="Block Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
#ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
path = OUTPUT_DIR / f"cv_timeseries_L{LAYER_TO_ANALYZE}_{MODEL}.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
