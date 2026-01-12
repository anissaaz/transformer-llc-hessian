import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig

from lm_hf_curvlinops import run_hessian_analysis


parser = argparse.ArgumentParser(description="specify which transformer section to compute metrics for")
parser.add_argument("-m", "--model", help="Pythia HF model id", default="EleutherAI/pythia-14m")

parser.add_argument("-l", "--layer", type=int, choices=range(0,12), help="model layer")
parser.add_argument("-H", "--head", type=int, default=None, help="attention head")

parser.add_argument("-s", "--size", type=int, default=500, help="Target Hessian size")

parser.add_argument("-c", "--components", nargs='+', 
                    choices=[
                        "embed_in",
                        "attention.qkv", 
                        "attention.q",   # just Q
                        "attention.k",     # just K
                        "attention.v",   # just V
                        "attention.out",  # Matches 'wo' / dense
                        "mlp.exp",     # Matches 'c_fc' / dense_h_to_4h
                        "mlp.cont",   # Matches 'c_proj' / dense_4h_to_h
                        "embed_out"
                    ], 
                    help="component of the layer")

args = parser.parse_args()

use_sequential_sampling = True

config = AutoConfig.from_pretrained(args.model)
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
hidden_size = config.hidden_size
head_dim = int(hidden_size / num_heads)


if args.head is not None:
    if args.head < 0 or args.head >= num_heads:
        parser.error(f"--head must be between 0 and {num_heads-1}")

def get_component(model, layer, component_name):
    """
    Args:
        model: The 'MyTransformer' instance passed from inside the loop.
    Returns:
        List of parameters to analyze.
    """
    
    hf_model = model.hf_model

    if layer is not None:
        block = hf_model.gpt_neox.layers[layer]
        
    if component_name == "embed_in":
        p = hf_model.gpt_neox.embed_in.weight
        return [p], torch.arange(p.numel())
    
    if component_name == "embed_out":
        p = hf_model.embed_out.weight
        return [p], torch.arange(p.numel())
        

    # --- ATTENTION ---    
    elif "attention" in component_name:
        if layer is None:
            parser.error("For 'attention' or 'mlp', you must also provide --layer")
            
        qkv_tensor = block.attention.query_key_value.weight     # shape [3*hidden_size, hidden_size]
        
        # map of all indices reshaped to 2D [Rows, Cols]
        all_indices = torch.arange(qkv_tensor.numel())
        indices_2d = all_indices.reshape(qkv_tensor.shape[0], qkv_tensor.shape[1])
        
        # offset with head slicing
        h_start = 0 if args.head is None else args.head * head_dim
        h_end = hidden_size if args.head is None else (args.head + 1) * head_dim
        
        if component_name == "attention.qkv":
            if args.head is None:
                 return [qkv_tensor], all_indices
            else:
                 # Construct list of indices for Q-head, K-head, V-head
                 q_slice = indices_2d[h_start:h_end, :].flatten()
                 k_slice = indices_2d[hidden_size + h_start : hidden_size + h_end, :].flatten()
                 v_slice = indices_2d[2*hidden_size + h_start : 2*hidden_size + h_end, :].flatten()
                 return [qkv_tensor], torch.cat([q_slice, k_slice, v_slice])
             
        elif component_name == "attention.q":
            subset = indices_2d[h_start:h_end, :]
            return [qkv_tensor], subset.flatten()
        
        elif component_name == "attention.k":
            subset = indices_2d[hidden_size + h_start : hidden_size + h_end, :]
            return [qkv_tensor], subset.flatten()
        
        elif component_name == "attention.v":
            subset = indices_2d[2*hidden_size + h_start : 2*hidden_size + h_end, :]
            return [qkv_tensor], subset.flatten()
        
        elif component_name == "attention.out":
            p = block.attention.dense.weight
            all_indices = torch.arange(p.numel())
            indices_2d = all_indices.reshape(p.shape[0], p.shape[1])
            
            subset = indices_2d[:, h_start:h_end]
            return [p], subset.flatten()
        
        #output_suffix = f"l{args.layer}_{args.component}".replace(".", "_")

    # --- MLP ---
    elif "mlp" in component_name:
        
        h_start = 0 if args.head is None else args.head * head_dim
        h_end = hidden_size if args.head is None else (args.head + 1) * head_dim
        
        if args.layer is None:
            parser.error("For MLP, you must provide --layer")

        if component_name == "mlp.exp":
            p = block.mlp.dense_h_to_4h.weight             # Corresponds to c_fc (dense_h_to_4h)
            all_indices = torch.arange(p.numel())
            indices_2d = all_indices.reshape(p.shape[0], p.shape[1])
            
            subset = indices_2d[:, h_start:h_end]
            return [p], subset.flatten()
        
        elif component_name == "mlp.cont":
            # Corresponds to c_proj (dense_4h_to_h)
            p = block.mlp.dense_4h_to_h.weight
            all_indices = torch.arange(p.numel())
            indices_2d = all_indices.reshape(p.shape[0], p.shape[1])
            
            subset = indices_2d[h_start:h_start+2, :]
            return [p], subset.flatten()
        
    raise ValueError(f"Unknown component: {component_name}")

        
#print(f"output suffix: {output_suffix}")


def param_selector(model, verbose=True):
    unique_tensors = []
    tensor_id_to_start = {}
    current_global_offset = 0
    
    component_metadata = []
    
    if verbose:
        print(f"Collecting params for: {args.components}")
    
    for comp in args.components:
        
        tensor_list, local_idx = get_component(model, args.layer, comp)
        tensor = tensor_list[0]
        t_id = id(tensor)
    
        if t_id not in tensor_id_to_start:
            unique_tensors.append(tensor)
            tensor_id_to_start[t_id] = current_global_offset
            
            start_global = current_global_offset
            current_global_offset += tensor.numel()
            
        
        else:
            start_global = tensor_id_to_start[t_id]
            
        #import ipdb; ipdb.set_trace()
        
        global_idx = local_idx + start_global
        
        component_metadata.append({
            "name": comp,
            "global_indices": global_idx,
            "count": len(global_idx)
        })
        
        
    total_requested_params = sum(data["count"] for data in component_metadata)
    final_indices_list = []
    
    component_counts = []
    
    print(f"Total parameters: {total_requested_params}")
    
    # sampling
    if total_requested_params > args.size:
        
        if verbose: print(f"Subsampling to {args.size}")
        
        for data in component_metadata:
            
            ratio = data["count"] / total_requested_params
            n_sample = int(ratio * args.size)       
            
            if use_sequential_sampling:
                
                # CASE A: just one head
                if args.head is not None:
                    
                    if verbose:
                        print(f"     -> Sample size of {data['name']}: {n_sample}")
                        
                    # Just take the first n_sample indices
                    sampled_indices = data["global_indices"][:n_sample]
                
                # CASE B: all heads
                else:
                    chunk_size_per_h = n_sample // num_heads
                    
                    # stride within global indices considering full head to move to next head
                    total_params_h_stride = data["count"] // num_heads
                    
                    if verbose: 
                        print(f"  -> {data['name']}: Sampling first {chunk_size_per_h} params from each of {num_heads} heads.")
                    
                    # temporary list to store slices
                    head_slices = []
                    
                    for h in range(num_heads):
                        start = h * total_params_h_stride
                        end = start + chunk_size_per_h
                        
                        # check whether head's bounds exceeded if n_sample is large
                        end = min(end, start + total_params_h_stride)
                        
                        head_slices.append(data["global_indices"][start:end])
                    
                    # indices for current component
                    sampled_indices = torch.cat(head_slices)
            
            else:
                # random parameters
                perm = torch.randperm(data["count"])
                sampled_indices = data["global_indices"][perm[:n_sample]]
            
            # list of all sampled component indices
            final_indices_list.append(sampled_indices)
            component_counts.append((data["name"], len(sampled_indices)))
            
            #import ipdb; ipdb.set_trace()
            
            #if verbose: print(f"  -> {data['name']}: {data['count']} params. Keeping {len(sampled_indices)}.")
            
    else:
        if verbose: print("Taking all parameters (no subsampling needed).")
        final_indices_list = [m['global_indices'] for m in component_metadata]
        for data in component_metadata:
            component_counts.append((data['name'], data['count']))
        
    if final_indices_list:
        #import ipdb; ipdb.set_trace()
        final_indices = torch.cat(final_indices_list)
        final_indices, _ = torch.sort(final_indices)
        return unique_tensors, final_indices.tolist(), component_counts
    
    return unique_tensors, [], []


suffix = f"L{args.layer}"
if args.head is not None:
    suffix += f"_H{args.head}"

if len(args.components) > 1:
    short_names = [c.replace("attention", "attn") for c in args.components]
    comp_str = "_".join(short_names)
    suffix += f"_{comp_str}"
else:
    suffix += f"_{args.components[0]}"

if use_sequential_sampling:
    suffix += "_sequ"
else:
    suffix += "_random"
        
print(f"Starting analysis for: {suffix}")
run_hessian_analysis(
    args.model, 
    param_selector=param_selector, 
    output_suffix=suffix
)