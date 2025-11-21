import os
import glob
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENT_DIR = "hessian-pile-batch8"


def load_layer_csvs(pattern, label):
    """
    pattern: glob pattern, e.g. 'hessian_metrics_layer*_mlp.csv'
    returns dict[layer_index] -> DataFrame
    """
    files = sorted(glob.glob(os.path.join(EXPERIMENT_DIR, pattern)))
    
    if not files:
        print(f"[{label}] No files matched pattern: {pattern}")
        return {}

    data = {}
    for f in files:
        # expect something like 'hessian_metrics_layer2_mlp.csv'
        m = re.search(r"layer(\d+)", f)
        if not m:
            print(f"[{label}] Could not parse layer from {f}, skipping")
            continue
        layer = int(m.group(1))
        df = pd.read_csv(f).sort_values("step")
        data[layer] = df
    print(f"[{label}] Loaded {len(data)} layers from {pattern}")
    return data


def plot_metric_across_layers(data_by_layer, metric, title, out_path):
    """
    data_by_layer: dict[layer] -> DataFrame
    metric: 'trace', 'max_eig', or 'stable_rank'
    """
    if not data_by_layer:
        print(f"No data to plot for {title}")
        return

    plt.figure(figsize=(9, 5))

    for layer, df in sorted(data_by_layer.items()):
        plt.plot(
            df["step"],
            df[metric],
            marker="o",
            linewidth=1.1,
            markersize=3,
            label=f"layer {layer}",
        )

    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)

    plt.xscale("log")
    # symlog: linear near 0, log away from 0, supports negative values
    plt.yscale("symlog", linthresh=10)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    out_dir = os.path.join(EXPERIMENT_DIR, "plots_layers")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- MLP ----------
    mlp_data = load_layer_csvs("hessian_metrics_layer*_mlp.csv", "MLP")

    plot_metric_across_layers(
        mlp_data,
        metric="trace",
        title="MLP: trace vs step (symlog)",
        out_path=os.path.join(out_dir, "mlp_trace_symlog.png"),
    )
    plot_metric_across_layers(
        mlp_data,
        metric="max_eig",
        title="MLP: max eigenvalue vs step (symlog)",
        out_path=os.path.join(out_dir, "mlp_max_eig_symlog.png"),
    )
    plot_metric_across_layers(
        mlp_data,
        metric="stable_rank",
        title="MLP: stable rank vs step (symlog)",
        out_path=os.path.join(out_dir, "mlp_stable_rank_symlog.png"),
    )

    # ---------- Attention (QKV weights) ----------
    attn_data = load_layer_csvs(
        "hessian_metrics_layer*attention_query_key_value_weight.csv",
        "Attention QKV",
    )

    plot_metric_across_layers(
        attn_data,
        metric="trace",
        title="Attention QKV: trace vs step (symlog)",
        out_path=os.path.join(out_dir, "attn_qkv_trace_symlog.png"),
    )
    plot_metric_across_layers(
        attn_data,
        metric="max_eig",
        title="Attention QKV: max eigenvalue vs step (symlog)",
        out_path=os.path.join(out_dir, "attn_qkv_max_eig_symlog.png"),
    )
    plot_metric_across_layers(
        attn_data,
        metric="stable_rank",
        title="Attention QKV: stable rank vs step (symlog)",
        out_path=os.path.join(out_dir, "attn_qkv_stable_rank_symlog.png"),
    )

    # ---------- Embedding input (embed_in) ----------
    embed_in_csv = os.path.join(EXPERIMENT_DIR, "hessian_metrics_embed_in.csv")
    if os.path.exists(embed_in_csv):
        df_embed = pd.read_csv(embed_in_csv).sort_values("step")

        # fake a "layer" key so we can reuse the same plotting code
        embed_data = {"embed_in": df_embed}

        plot_metric_across_layers(
            embed_data,
            metric="trace",
            title="embed_in: trace vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_in_trace_symlog.png"),
        )
        plot_metric_across_layers(
            embed_data,
            metric="max_eig",
            title="embed_in: max eigenvalue vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_in_max_eig_symlog.png"),
        )
        plot_metric_across_layers(
            embed_data,
            metric="stable_rank",
            title="embed_in: stable rank vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_in_stable_rank_symlog.png"),
        )
    else:
        print("No hessian_metrics_embed_in.csv found yet, skipping embed_in plots.")
        
    # ---------- Embedding output (embed_out) ----------
    embed_out_csv = os.path.join(EXPERIMENT_DIR, "hessian_metrics_embed_out.csv")
    if os.path.exists(embed_out_csv):
        df_embed_out = pd.read_csv(embed_out_csv).sort_values("step")

        embed_out_data = {"embed_out": df_embed_out}

        plot_metric_across_layers(
            embed_out_data,
            metric="trace",
            title="embed_out: trace vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_out_trace_symlog.png"),
        )
        plot_metric_across_layers(
            embed_out_data,
            metric="max_eig",
            title="embed_out: max eigenvalue vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_out_max_eig_symlog.png"),
        )
        plot_metric_across_layers(
            embed_out_data,
            metric="stable_rank",
            title="embed_out: stable rank vs step (symlog)",
            out_path=os.path.join(out_dir, "embed_out_stable_rank_symlog.png"),
        )
    else:
        print("No hessian_metrics_embed_out.csv found yet, skipping embed_out plots.")


if __name__ == "__main__":
    main()