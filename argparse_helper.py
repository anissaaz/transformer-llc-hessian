import argparse
from lm_hf_curvlinops import main as run_hessian_main

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute Hessian metrics for HuggingFace transformer models."
    )
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped",
                        help="HuggingFace model name or path.")
    parser.add_argument("--layer", type=int, default=5,
                        help="Transformer layer index to analyze.")
    parser.add_argument("--part", type=str, default="attention",
                        help="Component of the layer (e.g. attention, mlp).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--num-matvecs", type=int, default=5,
                        help="Number of Hutchinson samples for trace/Frobenius estimates.")
    return parser

def main():
    args = build_parser().parse_args()

    # You can either pass args into your existing main function
    # or (better) refactor lm_hf_curvlinops.py so main() accepts arguments.
    run_hessian_main(
        model_name=args.model,
        layer=args.layer,
        part=args.part,
        device=args.device,
        num_matvecs=args.num_matvecs
    )

if __name__ == "__main__":
    main()