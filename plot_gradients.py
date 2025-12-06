import torch
import glob
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GRADIENTS_FOLDER = "hessian-batch0-7/14m/gradients-embed_in"


def load_and_process_grads(folder_path):
    
    # Get all .pt files in the directory
    files = sorted(glob.glob(os.path.join(GRADIENTS_FOLDER, "grad_step_*.pt")))
    
    if not files:
        print("No files found!")
        return pd.DataFrame()

    records = []
    print(f"Found {len(files)} files. Processing...")
    
    for f in files:
        # Load to CPU to save RAM
        data = torch.load(f, map_location="cpu")
        
        #import ipdb; ipdb.set_trace()
        
        step = data["step"]
        grad_tensor = data["grads"][0].float()
        
        grad_norm = torch.norm(grad_tensor).item()
        
        records.append({
            "step" : step,
            "grad_norm" : grad_norm
        })
    
    df = pd.DataFrame(records).sort_values("step")
    return df

def add_learning_stage_lines():
    """Add the dashed lines for three learning phases."""
    stages = [16, 256, 2000]
    for s in stages:
        plt.axvline(s, linestyle="--", linewidth=1, alpha=0.7)
        
def plot_grads(df):
    plt.figure(figsize=(10, 6))
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(
            data=df,
            x="step",
            y="grad_norm",
            marker="o",
            markersize=4,
        )
    
    add_learning_stage_lines()
    plt.xscale("log")
    plt.title("Input Embedding Gradient Norm", fontsize=14)
    plt.xlabel("Training Step (Log Scale)", fontsize=12)
    plt.ylabel("L2 Norm", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    
    output_filename = "gradient_norms.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved to {os.path.abspath(output_filename)}")
    
if __name__ == "__main__":
    df = load_and_process_grads(GRADIENTS_FOLDER)
    if not df.empty:
        plot_grads(df)