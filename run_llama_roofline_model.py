# chooper1/PerformanceModeling/tree/aditya-updated
# script to model LLM performance on GPU
import argparse
from gpu_llama_roofline_model_support_functions import compute_llm_prompt_runtime_gpu_roofline_model, compute_llm_runtime_gpu_roofline_model


def main(args):
    """
    Run analytical model for an LLM
    """

    print(f"LLM Runtime with a sequence length of {args.sequence_length}")

    if args.input_prompt_length > 0:
        total_p, ffn_fc_p, mha_fc_p, mha_attn_p, nl_p, cl_p, l_ai_p, a_ai_p, agg_ai_p, total_flops_p = compute_llm_prompt_runtime_gpu_roofline_model(
            hidden_dim=args.hidden_dim,
            num_attn_heads=args.num_attn_heads,
            gqa_factor=args.gqa_factor,
            proj_factor=args.proj_factor,
            num_decoder_layers=args.num_decoder_layers,
            include_classifier=args.include_classifier,
            num_classes=args.num_classes,
            start_sequence_length=0,
            end_sequence_length=args.input_prompt_length,
            batch_size=args.batch_size
        )
    else:
        total_p, ffn_fc_p, mha_fc_p, mha_attn_p, nl_p, cl_p, l_ai_p, a_ai_p, agg_ai_p, total_flops_p = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    if args.sequence_length > 0:
        total_g, ffn_fc_g, mha_fc_g, mha_attn_g, nl_g, cl_g, l_ai_g, a_ai_g, agg_ai_g, total_flops_g = compute_llm_runtime_gpu_roofline_model(
            sequence_length=args.sequence_length,
            hidden_dim=args.hidden_dim,
            num_attn_heads=args.num_attn_heads,
            gqa_factor=args.gqa_factor,
            proj_factor=args.proj_factor,
            num_decoder_layers=args.num_decoder_layers,
            include_classifier=args.include_classifier,
            num_classes=args.num_classes,
            input_prompt_length=args.input_prompt_length,
            batch_size=args.batch_size
        )
    else:
        # If not generating, all generation costs are zero
        total_g, ffn_fc_g, mha_fc_g, mha_attn_g, nl_g, cl_g, l_ai_g, a_ai_g, agg_ai_g, total_flops_g = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # --- Aggregate and Print Results ---
    total = total_g + total_p
    total_flops = total_flops_g + total_flops_p
    
    # Avoid division by zero for percentage calculation if total is zero
    if total == 0:
        print("Total runtime is zero. No performance breakdown.")
        return
        
    ffn_fc = ffn_fc_g + ffn_fc_p
    mha_fc = mha_fc_g + mha_fc_p
    mha_attn = mha_attn_g + mha_attn_p
    nl = nl_g + nl_p
    cl = cl_g + cl_p

    print("\n--- Total Runtime Breakdown ---")
    print(f"Total Runtime: {total:.6f} s")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Performance: {(total_flops / total) / 1e9:.2f} GFLOP/s")
    print(f"  - FFN Linear Layers: {ffn_fc:.6f} s ({(ffn_fc / total) * 100:.2f}%)")
    print(f"  - MHA Linear Layers: {mha_fc:.6f} s ({(mha_fc / total) * 100:.2f}%)")
    print(f"  - MHA Attention: {mha_attn:.6f} s ({(mha_attn / total) * 100:.2f}%)")
    print(f"  - Nonlinear Ops: {nl:.6f} s ({(nl / total) * 100:.2f}%)")
    print(f"  - Classifier: {cl:.6f} s ({(cl / total) * 100:.2f}%)")

    if total_p > 0:
        print("\n--- Prefill (Prompt) Phase Analysis ---")
        print(f"Total: {total_p:.6f} s")
        print(f"Linear AI: {l_ai_p:.4f}")
        print(f"Attention AI: {a_ai_p:.4f}")
        print(f"Aggregate AI: {agg_ai_p:.4f}")

    if total_g > 0:
        print("\n--- Decode (Generation) Phase Analysis ---")
        print(f"Total: {total_g:.6f} s")
        print(f"Linear AI: {l_ai_g:.4f}")
        print(f"Attention AI: {a_ai_g:.4f}")
        print(f"Aggregate AI: {agg_ai_g:.4f}")


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments (default values are for Llama-3-8B-Instruct)
    parser.add_argument("--sequence_length", help="Sequence Length", type=int, default=2048)
    parser.add_argument("--hidden_dim", help="Hidden Dimension", type=int, default=4096)
    parser.add_argument("--num_attn_heads", help="Number of Attention Heads", type=int, default=32)
    parser.add_argument("--gqa_factor", help="Number of query head groups sharing the keys and values", type=int, default=4)  # 32/8 = 4
    parser.add_argument("--proj_factor", help="FFN Projection Factor", type=float, default=3.5)  # 14336/4096 = 3.5
    parser.add_argument("--num_decoder_layers", help="Number of Decoder Layers", type=int, default=32)
    parser.add_argument("--include_classifier", help="Whether to include the classifier", type=bool, default=True)
    parser.add_argument("--num_classes", help="Number of Classes", type=int, default=128256)  # vocab_size
    parser.add_argument("--activation_precision", help="Activation Precision", type=int, default=16)
    parser.add_argument("--input_prompt_length", help="Input Prompt Length", type=int, default=0)
    parser.add_argument("--batch_size", help="Batch Size", type=int, default=1)

    # Print version
    parser.add_argument("--version", action="version", version='0.1')

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parseArguments()
    main(args)