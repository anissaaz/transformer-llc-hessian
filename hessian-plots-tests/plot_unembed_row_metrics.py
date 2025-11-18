# plot_unembed_row_metrics.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def moving_avg(x, k):
    if k <= 1:
        return x
    return x.rolling(window=k, min_periods=1, center=False).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pythia_unembed_row_metrics.csv")
    ap.add_argument("--model_id", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Basic sanity prints
    print(df.head())
    print("\nUnique token_ids:", df["token_id"].unique())

    # Decode the token_id to check what word you're tracking
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    first_token_id = int(df["token_id"].iloc[0])
    print("Token for token_id:", first_token_id, "->", repr(tokenizer.decode([first_token_id])))

    # Optional smoothing
    df["wrow_max_eig_smooth"] = moving_avg(df["wrow_max_eig"], args.smooth)
    df["wrow_trace_smooth"]   = moving_avg(df["wrow_trace"],   args.smooth)

    # Plot (linear scale)
    plt.figure(figsize=(11, 6))
    plt.plot(df["step"], df["wrow_max_eig_smooth"], label="wrow_max_eig", linewidth=2)
    plt.plot(df["step"], df["wrow_trace_smooth"],   label="wrow_trace",   linewidth=1.5, linestyle="--")
    plt.xlabel("Training step")
    plt.ylabel("Value")
    plt.title("Unembedding-row Hessian (correct token) — linear scale")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("unembed_row_hessian_linear.png", dpi=200)
    print("Saved unembed_row_hessian_linear.png")

    # Plot (log scale)
    plt.figure(figsize=(11, 6))
    plt.semilogy(df["step"], df["wrow_max_eig_smooth"], label="wrow_max_eig", linewidth=2)
    plt.semilogy(df["step"], df["wrow_trace_smooth"],   label="wrow_trace",   linewidth=1.5, linestyle="--")
    plt.xlabel("Training step")
    plt.ylabel("Value (log scale)")
    plt.title("Unembedding-row Hessian (correct token) — log scale")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("unembed_row_hessian_log.png", dpi=200)
    print("Saved unembed_row_hessian_log.png")

if __name__ == "__main__":
    main()