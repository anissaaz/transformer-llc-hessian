import glob
import re
import os

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(".")
RUN_DIRS = sorted(BASE_DIR.glob("hessian-batch0-7/31m"))
MAX_STEP = 10000
OUTPUT_DIR = Path("hessian-batch0-7/31m/plots")

def load_attention_qkv_runs():
    """
    Returns one big DataFrame with columns:
    [run, layer, step, trace, max_eig, stable_rank]
    for all attention QKV csvs in all run dirs.
    """
    dfs = []
    for run_dir in RUN_DIRS:
        run_name = run_dir.name
        
        pattern = str(run_dir / "hessian_metrics_layer*attention_query_key_value_weight*.csv")
        for csv_path in glob.glob(pattern):
            m = re.search(r"layer(\d+)", os.path.basename(csv_path))
            layer = int(m.group(1))
            
            df = pd.read_csv(csv_path)
            df["run"] = run_name
            df["layer"] = layer
            dfs.append(df)
        
    attn_df = pd.concat(dfs, ignore_index=True)
    return attn_df

def load_embed_runs(kind="embed_in"):
    """
    kind: 'embed_in' or 'embed_out'
    Returns one big DataFrame with columns:
    [run, step, trace, max_eig, stable_rank]
    """
    dfs = []
    for run_dir in RUN_DIRS:
        run_name = run_dir.name
        
        pattern = str(run_dir / f"hessian_metrics_{kind}_*.csv")
        
        csv_path = glob.glob(pattern)[0]
        df = pd.read_csv(csv_path)
        df["run"] = run_name
        dfs.append(df)
    
    embed_df = pd.concat(dfs, ignore_index=True)
    return embed_df

def add_learning_stage_lines():
    """Add the dashed lines for three learning phases."""
    for s in [16, 256, 2000]:
        plt.axvline(s, linestyle="--", linewidth=1, alpha=0.7)
        
def plot_attention(attn_df: pd.DataFrame, out_dir="hessian-batch0-7"):
    out_dir = Path(out_dir)
    
    attn_df_limit = attn_df[attn_df["step"] <= MAX_STEP].copy()
    
    for metric in ["trace", "max_eig", "stable_rank"]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
           data=attn_df_limit,
           x="step", 
           y=metric,
           hue="layer",          # one line per layer
           errorbar="se",        # standard error
           estimator="mean",
           marker="o",           # show points
           markersize=4,
        )
        
        add_learning_stage_lines()
        plt.xscale("log")
        #plt.yscale("symlog", linthresh=10)
        plt.title(f"Attention QKV: {metric} vs step - linear")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        
        out_path = out_dir / "31m" / "plots" / f"attn_qkv_{metric}_linear.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"saved {out_path}")

def plot_embedding(embed_df, kind):
    """
    kind: 'embed_in' or 'embed_out'
    embed_df has columns [run, step, trace, max_eig, stable_rank]
    """
    #out_dir = Path(out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    embed_df_limit = embed_df[embed_df["step"] <= MAX_STEP].copy()

    for metric in ["trace", "max_eig", "stable_rank"]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=embed_df_limit,
            x="step",
            y=metric,
            errorbar="se",       # standard error
            estimator="mean",
            marker="o",
            markersize=4,
        )
        
        add_learning_stage_lines()
        plt.xscale("log")
        #plt.yscale("symlog", linthresh=10)
        plt.title(f"{kind}: {metric} vs step)") #\n(Mean ± SE over {len(RUN_DIRS)} seeds
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        
        out_path = OUTPUT_DIR /f"{kind}_{metric}_avg.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()

        plt.close()
        print(f"saved {out_path}")
        
def main():
    attn_df = load_attention_qkv_runs()
    embed_in_df  = load_embed_runs("embed_in")
    embed_out_df = load_embed_runs("embed_out")

    plot_attention(attn_df)
    plot_embedding(embed_in_df,  "embed_in")
    plot_embedding(embed_out_df, "embed_out")


if __name__ == "__main__":
    main()