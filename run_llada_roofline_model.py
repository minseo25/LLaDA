import argparse
from gpu_llada_roofline_model_support_functions import compute_llada_step_runtime_gpu_roofline_model

def main(args):
    """
    Run analytical model for a LLaDA-style model, supporting semi-autoregressive analysis.
    """
    # Validate arguments based on generate.py logic
    if args.sequence_length % args.block_length != 0:
        raise ValueError("Generation length (sequence_length) must be divisible by block_length.")
    if args.steps % (args.sequence_length // args.block_length) != 0:
        raise ValueError("Total steps must be divisible by the number of blocks.")

    num_blocks = args.sequence_length // args.block_length
    steps_per_block = args.steps // num_blocks

    total_runtime = 0
    total_flops = 0
    total_mops = 0

    print("="*50)
    print("LLaDA Theoretical Performance Analysis")
    print("="*50)
    if num_blocks > 1:
        print("Mode: Semi-Autoregressive")
    else:
        print("Mode: Non-Semi-Autoregressive")
    print(f"Prompt: {args.input_prompt_length}, Generation: {args.sequence_length}, Total Steps: {args.steps}")
    print(f"Blocks: {num_blocks}, Block Length: {args.block_length}, Steps per Block: {steps_per_block}\n")
    print("--- Block-by-Block Analysis ---")

    for i in range(num_blocks):
        block_num = i + 1
        # Sequence length grows with each block
        current_sequence_length = args.input_prompt_length + (block_num * args.block_length)
        
        # Calculate performance for a single pass with the current sequence length
        runtime_one_pass, flops_one_pass, mops_one_pass = compute_llada_step_runtime_gpu_roofline_model(
            total_sequence_length=current_sequence_length,
            hidden_dim=args.hidden_dim,
            num_attn_heads=args.num_attn_heads,
            proj_factor=args.proj_factor,
            num_decoder_layers=args.num_decoder_layers,
            batch_size=args.batch_size
        )
        
        # Accumulate totals
        total_runtime += runtime_one_pass * steps_per_block
        total_flops += flops_one_pass * steps_per_block
        total_mops += mops_one_pass * steps_per_block

        performance_gflops = (flops_one_pass / runtime_one_pass) / 1e9
        print(f"Block {block_num}: SeqLen={current_sequence_length}, Time/Step={runtime_one_pass*1000:.4f} ms, FLOPs={flops_one_pass:.2e}, Performance={performance_gflops:.2f} GFLOP/s, Steps={steps_per_block}")

    # Final aggregate arithmetic intensity
    final_aggregate_ai = total_flops / total_mops if total_mops > 0 else 0
    
    print("\n" + "-"*50)
    print("--- Total Generation Analysis ---")
    print(f"Total Estimated Runtime: {total_runtime * 1000:.4f} ms")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Average Performance: {(total_flops / total_runtime) / 1e9:.2f} GFLOP/s")
    print(f"Final Aggregate Arithmetic Intensity (AI): {final_aggregate_ai:.4f}\n")


def parseArguments():
    parser = argparse.ArgumentParser()

    # LLaDA/LLaMA-3 shared parameters
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--num_decoder_layers", type=int, default=32)
    
    # LLaDA specific parameters (MHA)
    parser.add_argument("--num_attn_heads", type=int, default=32)
    parser.add_argument("--proj_factor", type=float, default=3.0)

    # Experimental conditions
    parser.add_argument("--input_prompt_length", type=int, default=128)
    parser.add_argument("--sequence_length", help="Total generation length", type=int, default=512)
    parser.add_argument("--steps", help="Total number of sampling steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=512) # Default to non-semi-autoregressive
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()
    main(args)
