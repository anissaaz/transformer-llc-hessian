from pathlib import Path
import torch

CKPT_DIR = Path("/local/home/aziane/transformer-llc-hessian/neox-ckpt-70m")

optim_files = sorted(CKPT_DIR.glob("zero_pp_rank_*_mp_rank_*_optim_states.pt"))

print("Found optimizer shards:")
for f in optim_files:
    print(" -", f.name)

# Loop over all optimizer shards
for f in optim_files:
    print(f"\n=== Loading {f.name} ===")
    obj = torch.load(f, map_location="cpu")
    
    import ipdb; ipdb.set_trace();

    print("Top-level keys:", list(obj.keys()))

    # DeepSpeed nested optimizer state
    if "optimizer_state_dict" in obj:
        osd = obj["optimizer_state_dict"]
        print("Format: optimizer_state_dict in DeepSpeed format")
        print("keys:", list(osd.keys()))
        if "state" in osd:
            print("  number of params with state:", len(osd["state"]))
        continue

    print("Unknown format - print keys to inspect manually.")
    
osd = obj["optimizer_state_dict"]
base = osd["base_optimizer_state"]

# go inside list until it's a dict
while isinstance(base, list):
    print("base is a list, length:", len(base))
    base = base[0]

# base should be dict with "state" and "param_groups"
print("Final base type:", type(base))
print("base keys:", base.keys())

exp_avg_list = base["exp_avg"]
exp_avg_sq_list = base["exp_avg_sq"]

print("Number of exp_avg tensors:", len(exp_avg_list))
print("Number of exp_avg_sq tensors:", len(exp_avg_sq_list))

# peek at the first few tensors
for i, (ea, ea2) in enumerate(zip(exp_avg_list, exp_avg_sq_list)):
    # ea = exp_avg_list[i]
    # ea2 = exp_avg_sq_list[i]
    print(f"Element {i}: exp_avg = {ea.item()}, exp_avg_sq = {ea2.item()}")
    if i >= 2:
        break