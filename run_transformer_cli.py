from lm_hf_curvlinops import run_hessian_analysis

import argparse
parser = argparse.ArgumentParser(description="specify which transformer section to compute metrics for")
parser.add_argument("-m", "--model", help="Pythia HF model id", default="EleutherAI/pythia-14m")
parser.add_argument("-c", "--component", choices=["embed_in", "attention", "attention.query_key_value.weight", "attention.dense.weight", "mlp", "embed_out"], help="component of the layer")
parser.add_argument("-l", "--layer", type=int, choices=[0, 1, 2, 3, 4, 5], help="model layer")
args = parser.parse_args()

if args.component in ["attention", "attention.query_key_value.weight", "attention.dense.weight", "mlp"]:
    if args.layer is None:
        parser.error("For 'attention' or 'mlp', you must also provide --layer")
    part = f"hf_model.gpt_neox.layers.{args.layer}.{args.component}"
    output_suffix = f"layer{args.layer}_{args.component}".replace(".", "_")
    
elif args.component == "embed_in":
    if args.layer is not None:
        parser.error("For 'embed_in' don't provide --layer")
    part = f"hf_model.gpt_neox.{args.component}.weight"
    output_suffix = args.component
    
elif args.component == "embed_out":
    if args.layer is not None:
        parser.error("For 'embed_out' don't provide --layer")
    part = f"hf_model.{args.component}.weight"
    output_suffix = args.component
    
else:
    part = None
    output_suffix = "full_model"
    print("No layer/component specified - using full model")
    
print(f"The Hessian metrics will be computed for: {part or 'FULL MODEL'}")
#print(f"output suffix: {output_suffix}")


run_hessian_analysis(args.model, part=part, output_suffix=output_suffix)