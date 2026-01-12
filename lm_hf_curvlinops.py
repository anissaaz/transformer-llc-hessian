r"""Usage with Huggingface LLMs
===============================

This example demonstrates how to work with Huggingface (HF) language models.

As always, let's first import the required functionality.
Remember to run :code:`pip install -U transformers datasets`
"""

import scipy
import scipy.sparse.linalg
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import os

from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt

from collections import UserDict
from collections.abc import MutableMapping

from dataclasses import dataclass, asdict
from datasets import load_dataset

import torch.utils.data as data_utils
from torch import Tensor, bfloat16, eye, manual_seed, no_grad
from torch.nn import CrossEntropyLoss, Module
from transformers import (
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from huggingface_hub import list_repo_refs
from huggingface_hub import hf_hub_download

from curvlinops import HessianLinearOperator
from curvlinops import hutchinson_trace
from curvlinops import hutchinson_squared_fro
from curvlinops import hutchinson_diag
from curvlinops.submatrix import SubmatrixLinearOperator

import torch

# --- memory config ---

try:
    # New PyTorch 2.1+ API
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sdpa_kernel(SDPBackend.MATH)
except Exception:
    # Fallback for older versions
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class HessianMetrics:
    revision: str
    step: int
    trace: float
    #trace_std: float
    max_eig: float
    #max_eig_std: float
    stable_rank: float
    #stable_rank_std: float

# make deterministic
manual_seed(0)

# --------- Config ---------
#MODEL = "EleutherAI/pythia-70m-deduped"
EXPERIMENT_DIR = "hessian-batch0-63/14m"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
COMPUTE_GRADS = False
COMPUTE_HESSIAN = True              # Must be True for Clustering or Plot or Metrics to work
COMPUTE_HESSIAN_METRICS = False
PLOT_HESSIAN = True
CLUSTER_HESSIAN = False
MAX_LEN = 256

def get_step_tags(model_id: str):
    refs = list_repo_refs(model_id)
    step_refs = []
    for ref in list(refs.tags) + list(refs.branches):
        m = re.fullmatch(r"step(\d+)", ref.name)
        if m:
            step_refs.append((ref.name, int(m.group(1))))
    step_refs.sort(key=lambda x: x[1])
    return step_refs


# Model
# -----

class MyTransformer(Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model, revision) -> None:
        super().__init__()
        self.hf_model = AutoModelForCausalLM.from_pretrained(model, revision=revision)

        # grad only for last layer
        # for p in self.hf_model.parameters():
        #     p.requires_grad = False
            
        # head = self.hf_model.get_output_embeddings()
        
        # for p in head.parameters():
        #     p.requires_grad = True

    def forward(self, data: MutableMapping) -> Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, sequence_length, vocab_size)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attention = data["attention_mask"].to(device)
        outputs = self.hf_model(input_ids, attention_mask=attention)
        logits = outputs.logits
        
        return logits.reshape(logits.shape[0] * logits.shape[1], -1)

# with no_grad():
#     logits = model(batch)
#     print(f"Logits shape: {logits.shape}")

def batch_size_fn(x: MutableMapping):
    return x["input_ids"].shape[0]

# %%
#
# Curvlinops
# ----------


# def ce_loss(logits, labels):
#     return F.cross_entropy(logits, labels, reduction=ce_loss.reduction)

# ce_loss.reduction = 'mean'

# import ipdb; ipdb.set_trace()

#H = hessian @ eye(hessian.shape[0], device=params[0].device)


def top_k_evals(H, k=3, which="LM"):
    """
    Compute the top-k eigenvalues of a Curvlinops linear operator.

    Args:
        H: A Curvlinops linear operator (e.g., HessianLinearOperator)
        k: Number of eigenvalues to compute (default 3)
        which: Which eigenvalues to compute ('LM' = largest magnitude)

    Returns:
        A NumPy array of the top-k eigenvalues.
    """
    # Convert to a SciPy-compatible LinearOperator
    H_scipy = H.to_scipy()

    # Compute eigenvalues
    evals, _ = scipy.sparse.linalg.eigsh(H_scipy, k=k, which=which)
    evals = np.round(evals, 3)
    # print(f"Leading {k} Hessian eigenvalues: {evals}")
    return evals

def hutchinson_trace_estimate(H, num_matvecs=5):
    """
    Estimate the trace of a linear operator (e.g. Hessian) using Hutchinson’s method.

    Args:
        H: A Curvlinops linear operator or matrix (e.g. HessianLinearOperator).
        num_matvecs: Number of random vectors to use in the Monte Carlo estimate.
                     Higher values give more accurate estimates (default: 5).
        distribution: Random vector distribution ('rademacher' or 'normal').
                      Default is Rademacher (+1/-1 entries).

    Returns:
        trace_estimate: Estimated trace of H as a scalar (PyTorch tensor or float).
    """
    trace_estimate = hutchinson_trace(H, num_matvecs).item()
    # print(f"Trace estimate: {trace_estimate}")
    return trace_estimate

def stable_rank(H, num_matvecs=5):
    """
    Estimate stable rank(H) = ||H||_F^2 / (lambda_max(H))^2.
    - H: Curvlinops linear operator (e.g., HessianLinearOperator)
    - num_matvecs: Hutchinson samples for Frobenius^2 (more = less variance)
    - which: 'LA' (largest algebraic) or 'LM' (largest magnitude) for eigsh
    """
    # Frobenius norm squared via Hutchinson
    fro2 = hutchinson_squared_fro(H, num_matvecs).item()
    
    # Top eigenvalue (scalar)
    max_eig = float(top_k_evals(H, k=1)[0])
    
    stable_rank = fro2 / (max_eig**2)
    stable_rank = round(stable_rank, 3)
    return stable_rank

# def unigram_loss(logits, targets):
#     valid_targets = targets[targets != -100]                    # targets = batch["labels"].flatten() as defined in curvlinops operator
#     vocab_size = logits.shape[-1]
    
#     token_count = torch.bincount(valid_targets.cpu(), minlength=vocab_size)     # bincount runs faster on cpu
#     p = (token_count / token_count.sum()).to(logits.device)         # shape [V], later converted to 2-d tensor by PyTorch
    
#     # model distribution over vocab
#     log_q = F.log_softmax(logits, dim=-1)       # shape [B*T, V]
    
#     # KL divergence
#     loss = F.kl_div(log_q, p, reduction="batchmean")
    
#     return loss

def logabs(mat: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Computes log10(|x| + epsilon) for better Hessian visualization."""
    return mat.abs().clamp(min=epsilon).log10()

def run_hessian_analysis(model_name, param_selector=None, output_suffix="full_model"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    step_tags = get_step_tags(model_name)
    step_tags = [
        (rev, step)
        for rev, step in step_tags
        if step in {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 36000, 50000, 72000, 100000, 107000, 143000}
    ]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer_vocab_size = tokenizer.vocab_size
    
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split ="train[:1%]")
    #dataset = load_dataset("/pub/hofmann-scratch/datasets/the_pile_deduplicated")
    subset = dataset.shuffle(seed=0).select(range(0, 64))
    texts = [tokenizer.bos_token + ex["text"] + tokenizer.eos_token for ex in subset]

    batch = tokenizer(
        texts, 
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length=MAX_LEN,
        )
    
    batch["labels"] = batch["input_ids"][:,1:].clone()
    batch["input_ids"] = batch["input_ids"][:,:-1].clone()
    batch["attention_mask"] = batch["attention_mask"][:,:-1].clone()        # align with input_ids ?
    
    batch["labels"][batch["attention_mask"] == 0] = -100       # labels adjusted for ignore_index=-100 for CE loss
    
    results: list[HessianMetrics] = []
    printed_shape = False
    printed_stats = False
    
    # --- STORAGE FOR CLUSTERING ---
    clustering_data = [] # Stores (step, diagonal_tensor)
    clustering_counts = None # Will store the name mapping
    
    for rev, step in step_tags:
        print(f"==> {rev}")
        model = MyTransformer(tokenizer, model_name, revision=rev).to(device=device, dtype=bfloat16)
        model_vocab_size = model.hf_model.config.vocab_size
        
        model.hf_model.gradient_checkpointing_enable()
        
        component_counts = None
        
        if param_selector is not None:
            selector_output = param_selector(model, verbose=(not printed_shape))
            
            printed_stats = True
            
            if isinstance(selector_output, tuple):
                
                if len(selector_output) == 3:
                    params, indices, component_counts = selector_output
                    use_submatrix = True
                    
                elif len(selector_output) == 2:
                    params, indices = selector_output
                    use_submatrix = True
                    
                else:
                    params = selector_output[0]
                    use_submatrix = False
                
        else:
            params = list(model.parameters())
            use_submatrix = False
            
        # Ensure component_counts is populated even if param_selector is None (Full Model)
        if component_counts is None:
             component_counts = []
             for name, p in model.named_parameters():
                 if p.requires_grad:
                     component_counts.append((name, p.numel()))
                     
        # Save this mapping for the final clustering step (only need to do it once)
        if clustering_counts is None:
            clustering_counts = component_counts

        
        #import ipdb; ipdb.set_trace()
        
        # use this for unigram loss
        # loss_fn = lambda logits, targets: unigram_loss(logits, targets)
        # loss_fn.reduction = "mean"
                
        if COMPUTE_GRADS:
            grads_dir = os.path.join(EXPERIMENT_DIR, f"gradients-{output_suffix}")
            os.makedirs(grads_dir, exist_ok=True)
            print("   -> Computing Gradients...")
            model.zero_grad()
            
            # forward & backward pass
            logits = model(batch)
            target_labels = batch["labels"].flatten().to(logits.device)
            
            manual_loss = F.cross_entropy(logits, target_labels)
            manual_loss.backward()
        
            grads = []
            for param in params:
                if param.grad is not None:
                    # move to CPU to save GPU memory
                    grads.append(param.grad.detach().cpu().clone())
                else:
                    grads.append(None)
            
            # cleanup
            del logits, manual_loss
            model.zero_grad() 
            torch.cuda.empty_cache()
        
            save_path = os.path.join(grads_dir, f"grad_step_{step}.pt")
            torch.save({
                "step": step,
                "grads": grads
            }, save_path)
            print(f"      Saved gradients to {save_path}")
            
            del grads       # delete from RAM
        
        
        if COMPUTE_HESSIAN:
            
            # split batch for lower memory allocation
            micro_batches = []
            batch_size = batch["input_ids"].shape[0]
            
            for i in range(batch_size):
                
                # slice dictionary for i-th sample
                mb = {
                    k: v[i:i+1] 
                    for k, v in batch.items() 
                    if isinstance(v, torch.Tensor)
                }
                
                # slice and flatten targets
                mb_targets = mb["labels"].flatten()
                
                
                # add tuple (inputs, targets) to list
                micro_batches.append((mb, mb_targets))
            
            #import ipdb; ipdb.set_trace()
            
            hessian = HessianLinearOperator(
                model,
                CrossEntropyLoss(),
                params,
                #[(batch, batch["labels"].flatten())],
                micro_batches,
                check_deterministic=False,              # don't check randomness
                batch_size_fn=batch_size_fn,
            )
            
            if use_submatrix:
                print(f"   -> Wrapping in SubmatrixLinearOperator ({len(indices)}x{len(indices)})")
                subhessian = SubmatrixLinearOperator(hessian, indices, indices)
                            
            # print the Hessian shape only once
            if not printed_shape:
                print(f"Hessian shape: {hessian.shape}")
                
                print(f"Dataset Stats:")
                # calculate unique tokens used in specified batch
                valid_ids = batch["input_ids"][batch["attention_mask"] == 1]        # filter out padding tokens
                unique_count = torch.unique(valid_ids).numel()
                print(f"   Unique tokens: {unique_count} / {model_vocab_size} ({unique_count/model_vocab_size:.2%})")
                      
                printed_shape = True
                
            
            if CLUSTER_HESSIAN:
                print(f"   -> [Clustering] Computing diagonal for step {step}...")
                diag = hutchinson_diag(hessian, num_matvecs=5)
                
                # Move to CPU immediately to free GPU
                clustering_data.append(diag.cpu())
                
            if PLOT_HESSIAN and step in [0, 16, 256, 2000, 10000, 50000, 143000]:
                
                if use_submatrix:
                    target_hessian = subhessian
                    plot_dim_size = len(indices)
                    
                else:
                    target_hessian = hessian
                    full_size = sum(p.numel() for p in params)
                    plot_dim_size = min(500, full_size)
                
                print(f"   -> Generating Hessian plot for step {step}...")
                
                #num_params = sum(p.numel() for p in params)
                
                # --- Compute columns sequentially to avoid OOM ---
                H_plot_data = np.zeros((plot_dim_size, plot_dim_size), dtype=np.float32)
                
                batch_size = 128
                
                print(f"      Computing {plot_dim_size} columns in batches of {batch_size}...")
                
                for start_col in range(0, plot_dim_size, batch_size):
                    end_col = min(start_col + batch_size, plot_dim_size)
                    current_batch_width = end_col - start_col
                    
                    # Create a "slab" of the Identity matrix
                    # Shape: [Total_Params, Batch_Size]
                    # This creates vectors where v[i, 0]=1, v[i+1, 1]=1, etc.
                    V_batch = torch.zeros((plot_dim_size, current_batch_width), device=device, dtype=bfloat16)
                    
                    # fill diagonal with batch
                    for k in range(current_batch_width):
                        V_batch[start_col + k, k] = 1.0
                    
                    # compute Hessian-Matrix Product
                    # triggers one backward pass for 'current_batch_width' columns simultaneously  
                    try:
                        Hv_batch = target_hessian @ V_batch
                        
                        # Store results
                        # Hv_batch is [Rows, Batch_Cols] -> Map to H_plot_data[:, start:end]
                        H_plot_data[:, start_col:end_col] = Hv_batch.float().detach().cpu().numpy()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"      ! OOM with batch size {batch_size}. Clearing cache and retrying sequentially for this block.")
                            torch.cuda.empty_cache()
                            # Fallback logic could go here, or just crash if critical
                            raise e
                        else:
                            raise e
                    
                    # Cleanup
                    del V_batch, Hv_batch
                    torch.cuda.empty_cache()
                    
                
                # for i in range(plot_dim_size):
                #     # Create a single probe vector [Total_Params, 1]
                #     v = torch.zeros((plot_dim_size, 1), device=device, dtype=bfloat16)
                #     v[i, 0] = 1.0
                    
                #     # Compute Hessian-Vector Product
                #     # Triggers backward pass for just this one column
                #     Hv = target_hessian @ v
                    
                #     # Store result
                #     # Extract only the top-left part we care about and move to CPU
                #     H_plot_data[:, i] = Hv[:, 0].float().detach().cpu().numpy()
                    
                #     # Cleanup to keep VRAM flat
                #     del v, Hv
                #     torch.cuda.empty_cache()

                # --- Visualization ---
                # Log Abs transform to make features visible
                H_plot_data = np.log10(np.abs(H_plot_data) + 1e-6)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                min_val, max_val = H_plot_data.min(), H_plot_data.max()
                # fix min/max for comparability
                #FIXED_VMIN = -6.0
                #FIXED_VMAX = -1.0
                
                img = ax.imshow(H_plot_data, cmap="viridis", vmin=min_val, vmax=max_val)

                # DRAW DYNAMIC AXES
                if component_counts:
                    current_pos = 0
                    tick_locs = []
                    tick_labels = []
                    
                    for name, count in component_counts:
                        if count == 0: continue
                        
                        # Calculate center of this section for the label
                        center = current_pos + (count / 2)
                        tick_locs.append(center)
                        tick_labels.append(name)
                        
                        # Draw Divider Line at the end of this section
                        end_pos = current_pos + count
                        
                        # Don't draw line at the very end of image
                        if end_pos < H_plot_data.shape[0]:
                            # -0.5 puts the line exactly between pixels
                            ax.axhline(y=end_pos - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
                            ax.axvline(x=end_pos - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
                        
                        current_pos += count
                    
                    # Apply labels
                    ax.set_xticks(tick_locs)
                    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
                    ax.set_yticks(tick_locs)
                    ax.set_yticklabels(tick_labels, fontsize=9)
                    
                # Add secondary axes on Top and Right to show numeric indices (0, 100, 200...)
                # This works automatically because they inherit the 0..N limits from imshow
                sec_ax_x = ax.secondary_xaxis('top')
                #sec_ax_x.set_xlabel('Parameter Index', fontsize=10)
                sec_ax_x.tick_params(axis='x', labelsize=9)

                sec_ax_y = ax.secondary_yaxis('right')
                sec_ax_y.set_ylabel('Parameter Index', fontsize=10)
                sec_ax_y.tick_params(axis='y', labelsize=9)
                
                    
                ax.set_title(f"Hessian Sample: {output_suffix}\nStep {step}")
                cbar = fig.colorbar(img, ax=ax, shrink=0.8)
                cbar.set_label("log10(|Curvature|)")

                plot_filename = os.path.join(EXPERIMENT_DIR, f"hessian_step_{step}_{output_suffix}.png")
                plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"      Plot saved to {plot_filename}")
                
                # Clean up temp buffers
                del H_plot_data
                torch.cuda.empty_cache()
            
            if COMPUTE_HESSIAN_METRICS:
                
                # Compute metrics
                max_eig = top_k_evals(hessian, k=1)[0]
                trace_val = hutchinson_trace_estimate(hessian, num_matvecs=5)       # returns torch.Tensor
                stable_rank_val = stable_rank(hessian, num_matvecs=5)
                
                # # method error
                # R = 10  # number of method replicates

                # trace_vals = []
                # max_eig_vals = []
                # stable_rank_vals = []

                # for _ in range(R):
                #     # need to convert tensors to floats 
                #     trace_vals.append(float(hutchinson_trace(hessian, num_matvecs=50)))
                #     max_eig_vals.append(float(top_k_evals(hessian, k=1)[0]))
                #     stable_rank_vals.append(float(stable_rank(hessian, num_matvecs=50)))

                # trace_mean = float(np.mean(trace_vals))
                # trace_std  = float(np.std(trace_vals))

                # max_eig_mean = float(np.mean(max_eig_vals))
                # max_eig_std  = float(np.std(max_eig_vals))

                # stable_rank_mean = float(np.mean(stable_rank_vals))
                # stable_rank_std  = float(np.std(stable_rank_vals))
                
                print(f"max eig = {max_eig:.3f}, trace = {trace_val:.3f}, stable rank = {stable_rank_val:.3f}")
                
                results.append(HessianMetrics(
                    revision=rev,
                    step=step,
                    trace=trace_val,
                    #trace_std=trace_std,
                    max_eig=max_eig,
                    #max_eig_std=max_eig_std,
                    stable_rank=stable_rank_val,
                    #stable_rank_std=stable_rank_std,
                ))
            
            # Cleanup Hessian operator to free graph memory
            del hessian
            torch.cuda.empty_cache()
            
    if CLUSTER_HESSIAN and len(clustering_data) > 0:
        print("\n" + "="*40)
        print("Running K-Means Clustering on Parameter Trajectories")
        print("="*40)
        
        # 1. Prepare Data Matrix [Params x Steps]
        # Stack the collected diagonals
        X = torch.stack(clustering_data, dim=1).float().numpy()
        
        # Log-transform for better clustering (handles magnitude differences)
        X_log = np.log10(np.abs(X) + 1e-8)
        
        print(f"Data Shape: {X_log.shape} (Params x Steps)")
        
        # 2. Run Clustering
        n_clusters = 6
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=0)
        cluster_labels = kmeans.fit_predict(X_log)

        # 3. Map Clusters to Components using `clustering_counts`
        # We reconstruct the names based on the counts we saved
        
        # Create an array of names aligned with the parameters
        # e.g. ["embed", "embed", ... "layer0", "layer0"]
        all_names = []
        for name, count in clustering_counts:
            # --- A. Parse Layer Index ---
            # Search for "layers.X" pattern robustly
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                layer_idx = match.group(1)
                prefix = f"L{layer_idx}"
            else:
                prefix = "Emb/Head" # For embeddings or final layer norm

            # --- B. Identify Component & Split Q/K/V ---
            if "attention.query_key_value" in name:
                # This tensor contains Q, K, and V stacked.
                # We split the count into 3 equal chunks.
                chunk = count // 3
                
                # Handle potential rounding errors (though usually exact for transformers)
                remainder = count - (chunk * 3)
                
                all_names.extend([f"{prefix} Attn Q"] * chunk)
                all_names.extend([f"{prefix} Attn K"] * chunk)
                all_names.extend([f"{prefix} Attn V"] * (chunk + remainder))
                
            elif "attention.dense" in name:
                all_names.extend([f"{prefix} Attn Out"] * count)
                
            elif "mlp.dense_h_to_4h" in name:
                all_names.extend([f"{prefix} MLP Exp"] * count)
                
            elif "mlp.dense_4h_to_h" in name:
                all_names.extend([f"{prefix} MLP Cont"] * count)
                
            elif "embed_in" in name:
                all_names.extend(["Embed In"] * count)
                
            elif "embed_out" in name:
                all_names.extend(["Embed Out"] * count)
                
            else:
                # Fallback for LayerNorms, biases, etc.
                if "bias" in name:
                    all_names.extend(["Other Bias"] * count)
                elif "layernorm" in name:
                    all_names.extend([f"{prefix} LayerNorm"] * count)
                else:
                    all_names.extend(["Other"] * count)
        
        all_names = np.array(all_names)
        
        # 4. Create Statistics Table
        df = pd.DataFrame({
            "Component": all_names,
            "Cluster": cluster_labels
        })

        # Crosstab: Rows=Components, Cols=Clusters (as percentages)
        ct = pd.crosstab(df["Component"], df["Cluster"])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        
        print("\nCluster Distribution (% of parameters in each component):")
        print(ct_pct.round(1))
        
        # Optional: Save results
        ct_pct.to_csv(os.path.join(EXPERIMENT_DIR, "clustering_results.csv"))
        
        # Save centroids
        centroids_path = os.path.join(EXPERIMENT_DIR, "cluster_centroids.csv")
        
        # Get the list of steps we actually processed
        # (Extract just the step number from the tags list)
        processed_steps = [s[1] for s in step_tags]
        
        # Create a DataFrame for centroids
        # Rows = Clusters, Columns = Steps
        df_centroids = pd.DataFrame(
            kmeans.cluster_centers_, 
            columns=processed_steps
        )
        df_centroids.index.name = "Cluster"
        
        df_centroids.to_csv(centroids_path)
        print(f"Saved cluster centroids to {centroids_path}")
        
        
    if COMPUTE_HESSIAN_METRICS and results:
        csv_name = os.path.join(EXPERIMENT_DIR, f"hessian_metrics_{output_suffix}_0-7.csv")
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(csv_name, index=False, float_format="%.6f")
        print(f"\nSaved Hessian results to {csv_name}")
    
    
if __name__ == "__main__":
    run_hessian_analysis("EleutherAI/pythia-14m")