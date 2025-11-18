# scan_checkpoints.py
import os
import re
import math
import csv
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs


# --------- Config ---------
MODEL_ID = "EleutherAI/pythia-70m-deduped"   # pick a Pythia repo that exposes step revisions
PROMPT   = "The quick brown fox jumps over the lazy dog"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HF_HOME  = os.getenv("HF_HOME", None)        # optional: set to a fast local disk
# Limit or stride the steps to keep it fast on first run:
MAX_STEPS = None       # e.g. 10 to just do first 10; or None for all
STEP_STRIDE = 1        # e.g. 10 to sample every 10th step


@dataclass
class HiddenMetrics:
    revision: str
    step: int
    #max_eig: float
    #fro_sq: float
    #stable_rank: float
    #trace: float
    # wrow_max_eig: float
    # wrow_stable_rank: float
    # wrow_trace: float
    # token_id: int
    h_max_eig: float
    h_stable_rank: float
    h_trace: float
    h_dim: int


def get_step_tags(model_id: str):
    refs = list_repo_refs(model_id)
    step_refs = []
    for ref in list(refs.tags) + list(refs.branches):
        m = re.fullmatch(r"step(\d+)", ref.name)
        if m:
            step_refs.append((ref.name, int(m.group(1))))
    step_refs.sort(key=lambda x: x[1])
    return step_refs



@torch.no_grad()
def _tokenize(tokenizer, text, device):
    return tokenizer(text, return_tensors="pt").to(device)



# def _hessian_metrics_for_revision(model_id: str, revision: str, tokenizer, text: str) -> Metrics:
    """
    Loads one revision, computes Hessian wrt the last hidden state of the last token,
    and returns curvature metrics (max eigenvalue, Frobenius^2, stable rank, trace).
    """
    # Load model revision
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,        # keep autograd simple & stable
        low_cpu_mem_usage=True,
        cache_dir=HF_HOME,
    ).to(DEVICE)
    model.eval()

    # Prepare batch
    with torch.no_grad():
        inputs = _tokenize(tokenizer, text, DEVICE)

        # Forward with hidden states so we can take Hessian wrt h_last
        outputs = model(**inputs, output_hidden_states=True)
        # last token’s last hidden state: (1, d_model) -> requires grad
        h_last = outputs.hidden_states[-1][:, -1, :].detach().requires_grad_(True)

        # LM head is the output projection
        lm_head = model.get_output_embeddings()
        logits_last = lm_head(h_last)  # (1, vocab_size)

        # Use the real next token at that position as "target"
        target = inputs["input_ids"][:, -1]
        loss = F.cross_entropy(logits_last, target)
        
        rm = row_unembed_metrics(logits_last, h_last, target)

    # Build Hessian wrt h_last (d_model x d_model) via double autodiff
    grad = torch.autograd.grad(loss, h_last, create_graph=True)[0].squeeze(0)  # (d_model,)
    H_rows = []
    for g in grad:  # loop over d_model
        H_row = torch.autograd.grad(g, h_last, retain_graph=True)[0].squeeze(0)
        H_rows.append(H_row)
    H = torch.stack(H_rows)                        # (d_model, d_model)
    H = 0.5 * (H + H.T)                            # symmetrize (nicety)

    # Eigen-decomp (d_model is small enough for Pythia-70m/160m)
    eigvals = torch.linalg.eigvalsh(H).detach()
    max_eig = eigvals.max()
    fro_sq  = torch.sum(eigvals**2)                # == ||H||_F^2 for symmetric H
    trace   = eigvals.sum()

    stable_rank = (fro_sq / (max_eig**2 + 1e-12)).item()

    # Free a bit
    del model, outputs, H, grad, H_rows
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Parse step number from revision name
    m = re.fullmatch(r"step(\d+)", revision)
    step = int(m.group(1)) if m else -1

    return Metrics(
        revision=revision,
        step=step,
        max_eig=max_eig.item(),
        fro_sq=fro_sq.item(),
        stable_rank=stable_rank,
        trace=trace.item(),
    )


def _hidden_metrics_for_revision(model_id: str, revision: str, tokenizer, text: str) -> HiddenMetrics:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=HF_HOME,
    ).to(DEVICE)
    model.eval()

    with torch.no_grad():
        inputs = _tokenize(tokenizer, text, DEVICE)
        outputs = model(**inputs, output_hidden_states=True)

        h_last = outputs.hidden_states[-1][:, -2, :]  # (1, d_model)
        lm_head = model.get_output_embeddings()
        logits_last = lm_head(h_last)                 # (1, vocab_size)
        target = inputs["input_ids"][:, -1]           # last token

    import ipdb; ipdb.set_trace()
    rm = row_unembed_hessian_metrics_autograd(
        logits_last=logits_last,
        h_last=h_last,
        target=target,
        lm_head_weight=lm_head.weight
    )
    
    # --- Hessian wrt h (full-rank, d_model x d_model) ---
    h_metrics = hidden_hessian_metrics_autograd(
        h_last=h_last,
        target=target,
        lm_head_weight=lm_head.weight,
        #lm_head_bias=getattr(lm_head, "bias", None)
    )
    
    print(f"[H(h)] d_model={h_metrics['h_dim']}, "
        f"max_eig={h_metrics['h_max_eig']:.6g}, "
        f"trace={h_metrics['h_trace']:.6g}, "
        f"stable_rank={h_metrics['h_stable_rank']:.3f}")

    # cleanup
    del model, outputs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    step = int(re.fullmatch(r"step(\d+)", revision).group(1)) if re.fullmatch(r"step(\d+)", revision) else -1

    return HiddenMetrics(
        revision=revision,
        step=step,
        h_max_eig=h_metrics["h_max_eig"],
        h_stable_rank=h_metrics["h_stable_rank"],
        h_trace=h_metrics["h_trace"],
        h_dim=h_metrics["h_dim"],
    )
    
def row_unembed_hessian_metrics_autograd(
    logits_last: torch.Tensor,   # (1, vocab)
    h_last: torch.Tensor,        # (1, d_model)
    target: torch.Tensor,        # (1,) int64
    lm_head_weight: torch.Tensor # (vocab, d_model) to eval at current wy
):
    """
    Build a loss that depends ONLY on the correct-token row w_y, then
    take the Hessian wrt that row via autograd. Return eigen-based metrics.
    """
    y = int(target.item())                  # correct token id
    h = h_last.detach().squeeze(0)          # (d_model,)
    z_fixed = logits_last.detach().clone()  # baseline logits, frozen

    # initialize wy at the model's current unembedding row
    wy0 = lm_head_weight[y].detach().clone().requires_grad_(True)  # (d_model,)
    d_model = wy0.numel()
    print(f"[info] d_model = {d_model}  (unembedding-row dimension)")

    def loss_of_wy(wy):
        z = z_fixed.clone()
        z[0, y] = wy @ h             # ONLY y-th logit depends on wy
        return F.cross_entropy(z, target)

    # Full Hessian wrt wy (d_model x d_model)
    H = torch.autograd.functional.hessian(loss_of_wy, wy0)
    H = 0.5 * (H + H.T)              # symmetrize

    # Hessian metrics
    evals = torch.linalg.eigvalsh(H)
    lam_max = evals.max()
    trace   = evals.sum()
    fro_sq  = torch.sum(evals**2)
    stable_rank = (fro_sq / (lam_max**2 + 1e-12)).item()

    return {
        "wrow_max_eig": float(lam_max.item()),
        "wrow_trace":   float(trace.item()),
        "wrow_fro_sq":  float(fro_sq.item()),
        "wrow_stable_rank": stable_rank,
        "token_id": y,
    }

def hidden_hessian_metrics_autograd(
    h_last: torch.Tensor,        # (1, d_model)
    target: torch.Tensor,        # (1,) int64
    lm_head_weight: torch.Tensor, # (vocab, d_model)
    lm_head_bias: torch.Tensor | None = None
):
    """
    Build loss that depends on h (the last hidden state), then take Hessian wrt h.
    Returns eigen-based metrics of H_h (typically full-rank d_model x d_model).
    """
    h0 = h_last.detach().squeeze(0).requires_grad_(True)  # (d_model,)
    W = lm_head_weight                                   # (V, d_model)
    #b = lm_head_bias                                      # (V,) or None

    def loss_of_h(h):
        # z = W h + b; shape (1, V)
        z = F.linear(h.unsqueeze(0), W)               # (1, vocab)
        return F.cross_entropy(z, target)                # CE on last position

    # Full Hessian wrt h: (d_model x d_model)
    H = torch.autograd.functional.hessian(loss_of_h, h0)
    H = 0.5 * (H + H.T)  # numerical symmetrization

    # hessian metrics
    evals = torch.linalg.eigvalsh(H)
    lam_max = evals.max()
    trace   = evals.sum()
    fro_sq  = torch.sum(evals**2)
    stable_rank = float(fro_sq / (lam_max**2 + 1e-12))

    return {
        "h_max_eig":       float(lam_max.item()),
        "h_trace":         float(trace.item()),
        "h_fro_sq":        float(fro_sq.item()),
        "h_stable_rank":   stable_rank,
        "h_dim":           int(h0.numel()),
    }

def main():
    print(f"Using {DEVICE} | Model: {MODEL_ID}")
    step_tags = get_step_tags(MODEL_ID)
    if not step_tags:
        print("No step tags found for this repo. You can still analyze the final checkpoint.")
        step_tags = [("main", 0)]  # fall back to default branch

    # stride & cap
    step_tags = step_tags[::STEP_STRIDE]
    if MAX_STEPS is not None:
        step_tags = step_tags[:MAX_STEPS]

    # One tokenizer for all revisions
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_HOME)

    results: list[HiddenMetrics] = []
    for rev, step in step_tags:
        print(f"==> {rev} (step {step})")
        # try:
        #     m = _hessian_metrics_for_revision(MODEL_ID, rev, tokenizer, PROMPT)
        # except RuntimeError as e:
        #     # e.g., CUDA OOM; skip and continue
        #     print(f"  Skipped {rev}: {e}")
        #     continue
        # results.append(m)

        # try:
        #     m = _row_metrics_for_revision(MODEL_ID, rev, tokenizer, PROMPT)
        # except RuntimeError as e:
        #     print(f"  Skipped {rev}: {e}")
        #     continue
        # results.append(m)
        # print(f"  wrow_max_eig={m.wrow_max_eig:.6f}")
        
        try:
            m = _hidden_metrics_for_revision(MODEL_ID, rev, tokenizer, PROMPT)
        except RuntimeError as e:
            print(f"  Skipped {rev}: {e}")
            continue
        results.append(m)
        print(f"  h_max_eig={m.h_max_eig:.6f}  |  h_trace={m.h_trace:.6f}")
        

    # # Save CSV
    # out_csv = "pythia_hessian_metrics.csv"
    # with open(out_csv, "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(["revision", "step", "max_eig", "fro_sq", "stable_rank", "trace"])
    #     for m in results:
    #         w.writerow([m.revision, m.step, m.max_eig, m.fro_sq, m.stable_rank, m.trace])

    # print(f"\nSaved {len(results)} rows to {out_csv}")
    
    # Save CSV for unembedding
    # out_csv_rows = "pythia_unembed_row_metrics.csv"
    # with open(out_csv_rows, "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(["revision", "step", "wrow_max_eig", "wrow_stable_rank", "wrow_trace", "token_id"])
    #     for m in results:
    #         w.writerow([m.revision, m.step, m.wrow_max_eig, m.wrow_stable_rank, m.wrow_trace, m.token_id])
            
    # print(f"Saved {len(results)} rows to {out_csv_rows}")
    
    
    # Save CSV for unembedding
    out_csv = "pythia_hidden_metrics.csv"
    with open(out_csv, "w", newline="") as f:
         w = csv.writer(f)
         w.writerow(["revision", "step", "h_max_eig", "h_stable_rank", "h_trace", "h_dim"])
         for m in results:
             w.writerow([m.revision, m.step, m.h_max_eig, m.h_stable_rank, m.h_trace, m.h_dim])
            
    print(f"Saved {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()