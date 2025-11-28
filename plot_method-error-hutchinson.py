import glob
import re
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("mean-std-hessian-batch0-7")
RUN_DIRS = sorted(BASE_DIR.glob("mean-std-embedin-attn-50matvec"))

PLOTS_DIR = BASE_DIR / "mean-std-embedin-attn-50matvec" / "plots_50matvec_mean_std"

def load_attention_qkv_runs():
    dfs = []
    for run_dir in RUN_DIRS:
        run_name = run_dir.name
        pattern = str(run_dir / "mean_std_hessian_metrics_layer*_attention_query_key_value_weight_0-7.csv")
        
        for csv_path in glob.glob(pattern):
            m = re.search(r"layer(\d+)", os.path.basename(csv_path))
            layer = int(m.group(1))
            
            df = pd.read_csv(csv_path)
            df["run"] = run_name
            df["component"] = f"attn_L{layer}"
            dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def load_embed_runs(kind="embed_in"):
    dfs = []
    for run_dir in RUN_DIRS:
        run_name = run_dir.name
        pattern = str(run_dir / f"mean_std_hessian_metrics_{kind}_*.csv")
        matches = glob.glob(pattern)
        
        csv_path = matches[0]
        
        df = pd.read_csv(csv_path)
        df["run"] = run_name
        df["component"] = kind
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def add_learning_stage_lines():
    for s in [16, 256, 2000]:
        plt.axvline(s, linestyle="--", linewidth=1, alpha=0.6)

def plot_components(df, out_dir: Path, prefix=""):
    df_limit = df[df["step"] <= 1e4].copy()
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("trace", "trace_std"),
        ("max_eig", "max_eig_std"),
        ("stable_rank", "stable_rank_std"),
    ]

    components = sorted(df_limit["component"].unique())

    for metric, std_col in metrics:
        plt.figure(figsize=(10, 6))

        for comp in components:
            df_comp = df_limit[df_limit["component"] == comp]

            # sort by step (important for log plots)
            df_comp = df_comp.sort_values("step")

            plt.errorbar(
                df_comp["step"],
                df_comp[metric],
                yerr=df_comp[std_col],
                fmt="o-",
                markersize=4,
                capsize=3,
                label=comp,
                alpha=0.8
            )

        add_learning_stage_lines()
        plt.xscale("log")
        #plt.yscale("symlog", linthresh=10)

        plt.xlabel("Training step")
        plt.ylabel(metric)
        plt.title(f"{metric} (mean ± method std) - linear")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{prefix}{metric}_linear.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

def main():
    attn_df = load_attention_qkv_runs()
    embed_in_df = load_embed_runs("embed_in")
    #embed_out_df = load_embed_runs("embed_out")
    
    plot_components(attn_df, out_dir=PLOTS_DIR, prefix="attn_")
    plot_components(embed_in_df, out_dir=PLOTS_DIR, prefix="embed_in_")
    #plot_components(embed_out_df, out_dir=PLOTS_DIR, prefix="embed_out_")

if __name__ == "__main__":
    main()