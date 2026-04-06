import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

# --- available models with optimizer state checkpoints ---
# 70m-seed1
# 410m-seed1
# 1b
# 2.8b
# 6.9b

MODEL = "70m-seed1"
REPO_ID = f"EleutherAI/neox-ckpt-pythia-{MODEL}"
STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 36000, 50000, 72000, 100000, 143000]
BASE_SAVE_DIR = Path("./extracted_adam_states")
BASE_SAVE_DIR.mkdir(exist_ok=True)

def recover_optimizer_diagonal(ckpt_dir: Path):
    """
    Reconstruct per-parameter exp_avg_sq (Adam v_t) from DeepSpeed ZeRO-3 
    optimizer shards in ckpt_dir.

    Returns:
    dict  {param_name: tensor of shape matching original parameter shape}
    -------
    Weight-tied parameters (e.g. final_linear / LM head) appear in
    param_shapes metadata but have no optimizer state. They are skipped
    cleanly — this is expected and not an error.
    """
    # Load metadata from Rank 0 to get parameter shapes/names
    print("Loading metadata from Rank 0...")
    rank0_file = ckpt_dir / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
    rank0_obj = torch.load(rank0_file, map_location="cpu", weights_only=False)
    
    # Get the map of {param_name: torch.Size([...])}
    # This tells us how to "slice" the giant flattened vector later
    param_shapes = rank0_obj["param_shapes"]
    print(f"Found {len(param_shapes)} parameter definitions in metadata.")

    # --- Sort all shards by integer rank ---
    shard_files = sorted(
        ckpt_dir.glob("zero_pp_rank_*_mp_rank_00_optim_states.pt"), 
        key=lambda x: int(x.name.split('rank_')[1].split('_')[0])
    )
  
    # --- Stitch flat exp_avg_sq across all shards ---
    num_shards = len(shard_files)
    
    print(f"\nLoading {num_shards} shards to stitch the flattened buffer...")
    
    shards = []

    for f in tqdm(shard_files, desc="   shards", leave=False):
        obj = torch.load(f, map_location="cpu", weights_only=False)
        base_state = obj["optimizer_state_dict"]["base_optimizer_state"][0][0]
        shard_tensor = base_state["exp_avg_sq"]
        shards.append(shard_tensor)

    # Concatenate into one massive 1D tensor
    full_flat = torch.cat(shards)
    buffer_size = full_flat.shape[0]
    print(f"\nTotal flattened size: {buffer_size}")
    
    # Verify total expected elements vs actual buffer size upfront.
    # Weight-tied params (e.g. final_linear / LM head) appear in param_shapes
    # but have no optimizer state — the buffer legitimately ends before them.
    total_expected = sum(s.numel() for s in param_shapes.values())
    if total_expected != buffer_size:
        missing = total_expected - buffer_size
        print(f"  [info] param_shapes total={total_expected:,}, buffer={buffer_size:,} "
              f"— {missing:,} elements have no optimizer state (weight-tied, expected).")

    # --- Slice back into per-parameter tensors ---
    reconstructed = {}
    skipped       = []
    current_idx = 0
    
    print("Slicing flattened buffer into parameters...")
    for name, shape in param_shapes.items():
        num_elements = shape.numel()
        
        # Check if enough data left
        if current_idx + num_elements > full_flat.shape[0]:
            skipped.append(name)
            continue
            
        # Slice the chunk
        flat_param = full_flat[current_idx : current_idx + num_elements]
        
        # Reshape it to original parameter shape (e.g., [512, 512])
        reconstructed[name] = flat_param.view(shape).clone()
        
        current_idx += num_elements
        
    if skipped:
        print(f"  [info] skipped {len(skipped)} weight-tied param(s): {skipped}")

    print(f"Success! Reconstructed {len(reconstructed)} parameters.")
    return reconstructed

# ---------------------------------------------------------------------------
# Main loop — accumulate all steps, save once
# ---------------------------------------------------------------------------

# trajectory dict schema:
# {
#   "model":  "70m-seed1",
#   "steps":  [1, 2, 4, ..., 143000],           # list of ints, length S
#   "states": {                                  # one entry per successful step
#       1:      {"0.word_embeddings.weight": tensor, ...},
#       2:      {...},
#       ...
#       143000: {...},
#   }
# }

trajectory = {
    "model":  MODEL,
    "steps":  [],
    "states": {},
}

for step in STEPS:
    print(f"\n{'='*50}\nStep {step}\n{'='*50}")
    temp_dir = Path(f"./temp_ckpt_step{step}")

    try:
        # Download optimizer states for this branch
        snapshot_download(
            repo_id=REPO_ID,
            revision=f"step{step}",
            allow_patterns=["*optim_states.pt"],
            local_dir=temp_dir
        )
        
        state = recover_optimizer_diagonal(temp_dir)
        
        trajectory["steps"].append(step)
        trajectory["states"][step] = state
        
        print(f"  OK — {len(state)} params recovered")
        
    except Exception as e:
        print(f"Failed at step {step}: {e}")
        
    finally:
        # Delete the massive raw HF checkpoint shards
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up raw files for step {step}\n")


# save entire trajectory as one file
save_path = BASE_SAVE_DIR / f"adam_second_moment_trajectory_{MODEL}.pt"
torch.save(trajectory, save_path)
print(f"\nSaved full trajectory ({len(trajectory['steps'])} steps) -> {save_path}")

# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

test_key = "0.word_embeddings.weight"
last_step = trajectory["steps"][-1]

if test_key in trajectory["states"][last_step]:
    
    t = trajectory["states"][last_step][test_key]
    print(f"\nSanity check [{test_key}] at step {last_step}:")
    print(f"  shape : {t.shape}")
    print(f"  mean  : {t.mean().item():.6e}")
    print(f"  std   : {t.std().item():.6e}")
    print(f"  min   : {t.min().item():.6e}")
    print(f"  max   : {t.max().item():.6e}")