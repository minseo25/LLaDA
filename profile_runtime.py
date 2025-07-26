import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from generate import generate as generate_llada

# --- Global Settings ---
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
LLAMA_MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLADA_MODEL_ID = 'GSAI-ML/LLaDA-8B-Instruct'

# --- Helper Functions ---
def get_longbench_prompts(tokenizer, num_prompts=5):
    """
    Load long context prompts from LongBench-v2 (Single-Document QA)
    that have a context longer than 4096 tokens.
    """
    print("Loading LongBench-v2 dataset for long context prompts...")
    # Load the 'train' split of LongBench-v2 dataset
    dataset = load_dataset("THUDM/LongBench-v2", split="train", cache_dir="/rscratch/minseokim/hf_dataset")
    
    # Filter data for "Single-Document QA" domain
    print("Filtering for 'Single-Document QA' domain...")
    filtered_dataset = dataset.filter(lambda example: example['domain'] == 'Single-Document QA')
    
    long_samples = []
    print("Searching for samples with context > 4096 tokens...")
    for sample in filtered_dataset:
        context = sample['context']
        # Check if context exists and token length exceeds 4096
        if context and len(tokenizer(context)['input_ids']) > 4096:
            # Use only context since it will be truncated later anyway
            long_samples.append(context)
            # Stop when we have enough samples to improve efficiency
            if len(long_samples) >= num_prompts:
                break

    if len(long_samples) < num_prompts:
        raise ValueError(
            f"Could not find enough samples ({num_prompts}) with >4K token context "
            f"in 'Single-Document QA' domain. Found only {len(long_samples)}."
        )

    print(f"Found and selected {len(long_samples)} prompts.")
    return long_samples

def load_model_and_tokenizer(model_id, model_class):
    """Load the model and tokenizer."""
    print(f"\nLoading model: {model_id}...")
    model = model_class.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"{model_id} loaded successfully.")
    return model, tokenizer

def synchronize_and_time():
    """Helper function to measure GPU time accurately."""
    if 'cuda' in DEVICE:
        torch.cuda.synchronize()
    return time.time()

# --- 1. Prefill Cost Analysis ---
def profile_llama_prefill_long_context(model, tokenizer, prompts, truncation_lengths):
    """
    Measure the prefill runtime of Llama-3 for various truncated lengths
    of long context prompts, and average the results.
    """
    print("\n--- Profiling Llama-3 Prefill Cost on Long Contexts ---")
    all_results = []
    
    for length in tqdm(truncation_lengths, desc="Testing Truncation Lengths"):
        runtimes_for_length = []
        for prompt_text in prompts:
            # Tokenize and truncate
            input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
            if input_ids.shape[1] < length:
                continue # Skip if prompt is shorter than target length
            
            truncated_ids = input_ids[:, :length].to(DEVICE)
            
            with torch.no_grad():
                _ = model(truncated_ids) # Warm-up
            
            start_time = synchronize_and_time()
            with torch.no_grad():
                _ = model(truncated_ids)
            end_time = synchronize_and_time()
            
            runtimes_for_length.append((end_time - start_time) * 1000)

        if runtimes_for_length:
            avg_runtime_ms = np.mean(runtimes_for_length)
            all_results.append({'prompt_length': length, 'avg_runtime_ms': avg_runtime_ms})
    
    return pd.DataFrame(all_results)

# --- 2. Unit Generation Cost Analysis ---
def profile_llama_decode(model, tokenizer, base_prompt, prompt_lengths_to_test, max_gen_length):
    """
    Measure Llama-3 decode runtime per token for specific prompt lengths.
    """
    print("\n--- Profiling Llama-3 Decode Cost per Token ---")
    all_results = {}
    
    for p_len in tqdm(prompt_lengths_to_test, desc="Testing Prompt Lengths"):
        prompt_results = []
        
        # Truncate prompt and tokenize
        input_ids = tokenizer(base_prompt, return_tensors='pt').input_ids[:, :p_len].to(DEVICE)

        print(f"\nProfiling for prompt length: {input_ids.shape[1]}")

        # Prefill step
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        # Decode step, measuring each token
        for i in range(max_gen_length):
            start_time = synchronize_and_time()
            with torch.no_grad():
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            end_time = synchronize_and_time()
            
            duration_ms = (end_time - start_time) * 1000
            prompt_results.append({'gen_length': i + 1, 'runtime_ms': duration_ms})
            
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            
            if (next_token == tokenizer.eos_token_id).all():
                print(f"EOS token generated at step {i+1}. Stopping generation for this prompt.")
                break
        
        all_results[p_len] = pd.DataFrame(prompt_results)
        
    return all_results

def profile_llada_generation(model, tokenizer, base_prompt, prompt_lengths_to_test, gen_lengths_to_test):
    """
    Measure LLaDA total runtime and average step cost for various prompt and generation lengths.
    """
    print("\n--- Profiling LLaDA Total Runtime and Step Cost ---")
    all_results = []

    for p_len in tqdm(prompt_lengths_to_test, desc="Testing Prompt Lengths"):
        # Truncate prompt and tokenize for LLaDA's chat template
        base_input_ids = tokenizer(base_prompt, return_tensors='pt').input_ids[:, :p_len]
        # Then decode and apply chat template
        truncated_prompt_text = tokenizer.decode(base_input_ids[0])
        m = [{"role": "user", "content": truncated_prompt_text}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input, return_tensors='pt').input_ids.to(DEVICE)
        
        print(f"\nProfiling for prompt length: {input_ids.shape[1]}")

        for gen_len in tqdm(gen_lengths_to_test, desc="Testing Gen Lengths", leave=False):
            steps = max(1, gen_len // 2)
            block_length = gen_len # don't use semi-autoregressive sampling
            
            # Warm-up
            with torch.no_grad():
                _ = generate_llada(model, input_ids, steps=steps, gen_length=gen_len, block_length=block_length)

            # Measurement
            start_time = synchronize_and_time()
            with torch.no_grad():
                _ = generate_llada(model, input_ids, steps=steps, gen_length=gen_len, block_length=block_length)
            end_time = synchronize_and_time()
            
            total_duration_ms = (end_time - start_time) * 1000
            avg_step_ms = total_duration_ms / steps if steps > 0 else 0
            
            all_results.append({
                'prompt_length': input_ids.shape[1],
                'gen_length': gen_len,
                'total_runtime_ms': total_duration_ms,
                'avg_step_ms': avg_step_ms
            })

    return pd.DataFrame(all_results)

# --- 3. End-to-End Latency Analysis ---
def profile_llama_e2e_latency(model, tokenizer, base_prompt, prompt_lengths_to_test, gen_lengths_to_test):
    """
    Measure the end-to-end latency of Llama-3, splitting prefill and decode time.
    """
    print(f"\n--- Profiling Llama-3 End-to-End Latency ---")
    all_results = []

    for p_len in tqdm(prompt_lengths_to_test, desc="Testing Prompt Lengths"):
        # Truncate prompt and tokenize
        input_ids = tokenizer(base_prompt, return_tensors='pt').input_ids[:, :p_len].to(DEVICE)
        
        print(f"\nProfiling for prompt length: {input_ids.shape[1]}")

        for gen_len in tqdm(gen_lengths_to_test, desc="Testing Gen Lengths", leave=False):
            # Warm-up run
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            
            # 1. Measure Prefill time
            start_prefill = synchronize_and_time()
            with torch.no_grad():
                _ = model(input_ids, use_cache=True)
            end_prefill = synchronize_and_time()
            prefill_ms = (end_prefill - start_prefill) * 1000

            # 2. Measure total generation time
            start_total = synchronize_and_time()
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            end_total = synchronize_and_time()
            total_ms = (end_total - start_total) * 1000
            
            # 3. Calculate Decode time
            decode_ms = max(0, total_ms - prefill_ms)

            all_results.append({
                'prompt_length': input_ids.shape[1],
                'gen_length': gen_len,
                'prefill_ms': prefill_ms,
                'decode_ms': decode_ms,
                'total_runtime_ms': total_ms
            })
            
    return pd.DataFrame(all_results)

# --- Main function ---
def main():
    """Main execution function"""
    if DEVICE == 'cpu':
        print("="*80 + "\n!! Warning: Running on CPU is not feasible for 8B models. Exiting. !!\n" + "="*80)
        return

    # --- Experiment Parameters (New Plan) ---
    LONG_PROMPT_COUNT = 5
    # Llama-3 Prefill: test up to 4k only
    LLAMA_TRUNCATION_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # Llama-3 Decode/E2E: test up to 4k only
    LLAMA_PROMPT_LENGTHS_FOR_TEST = [128, 1024, 2048, 4096]
    
    # LLaDA Generation: test up to 2k
    LLADA_PROMPT_LENGTHS_FOR_TEST = [128, 1024, 2048]
    
    GEN_LENGTHS_FOR_TEST = [32, 64, 128, 256, 512, 1024, 2048]
    MAX_DECODE_LENGTH = 2048

    # --- Data Loading ---
    temp_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    # Now we can reliably get the longest prompts
    long_context_prompts = get_longbench_prompts(temp_tokenizer, LONG_PROMPT_COUNT)
    base_prompt_for_tests = long_context_prompts[0] # The longest prompt is selected
    del temp_tokenizer

    # --- Storages for Results ---
    llama_prefill_df = pd.DataFrame()
    llama_decode_results = {}
    llama_e2e_df = pd.DataFrame()
    llada_generation_df = pd.DataFrame()

    # --- 1. Llama-3 Analysis (Up to 4K) ---
    llama_model, llama_tokenizer = load_model_and_tokenizer(LLAMA_MODEL_ID, AutoModelForCausalLM)
    
    llama_prefill_df = profile_llama_prefill_long_context(llama_model, llama_tokenizer, long_context_prompts, LLAMA_TRUNCATION_LENGTHS)
    llama_decode_results = profile_llama_decode(llama_model, llama_tokenizer, base_prompt_for_tests, LLAMA_PROMPT_LENGTHS_FOR_TEST, MAX_DECODE_LENGTH)
    # E2E test is performed within Llama's context limit for safety
    llama_e2e_df = profile_llama_e2e_latency(llama_model, llama_tokenizer, base_prompt_for_tests, LLAMA_PROMPT_LENGTHS_FOR_TEST, GEN_LENGTHS_FOR_TEST)
    
    del llama_model, llama_tokenizer
    if 'cuda' in DEVICE: torch.cuda.empty_cache()

    # --- 2. LLaDA Analysis (Up to 4K) ---
    llada_model, llada_tokenizer = load_model_and_tokenizer(LLADA_MODEL_ID, AutoModel)

    llada_generation_df = profile_llada_generation(llada_model, llada_tokenizer, base_prompt_for_tests, LLADA_PROMPT_LENGTHS_FOR_TEST, GEN_LENGTHS_FOR_TEST)
    
    del llada_model, llada_tokenizer
    if 'cuda' in DEVICE: torch.cuda.empty_cache()

    # --- 3. Print All Results ---
    pd.set_option('display.width', 150)
    pd.set_option('display.max_rows', 100)
    print("\n\n" + "="*28 + " FINAL ANALYSIS RESULTS " + "="*28)

    # --- Analysis 1: Prefill Cost ---
    print("\n\n" + "-"*80)
    print("\n[ Analysis 1: Llama-3 Prefill Cost vs. Prompt Length ]")
    print("Average prefill cost (ms) across 5 long-context samples.")
    print(llama_prefill_df.to_string(index=False))

    # --- Analysis 2: Unit Generation Cost ---
    print("\n\n" + "-"*80)
    print("\n[ Analysis 2.1: Llama-3 Per-Token Decode Cost ]")
    print("Time (ms) to generate each subsequent token for different initial prompt lengths.")
    for p_len, df in llama_decode_results.items():
        print(f"\n--- Prompt Length: {p_len} tokens ---")
        # Display summary for brevity
        print(df.head())
        if len(df) > 10:
            print("...")
            print(df.tail())

    print("\n\n" + "-"*80)
    print("\n[ Analysis 2.2: LLaDA Total Runtime & Average Step Cost ]")
    print("Total runtime and average cost per step (ms) for different prompt and generation lengths.")
    for p_len in LLADA_PROMPT_LENGTHS_FOR_TEST:
         # LLaDA prompt length may vary slightly due to chat template, so use actual measured values
        # Use wider range for longer prompts
        range_width = max(30, p_len // 100)  # Wider range for longer prompts
        actual_prompt_len = llada_generation_df[llada_generation_df['prompt_length'].isin(range(p_len, p_len + range_width))]['prompt_length'].iloc[0]
        print(f"\n--- Prompt Length: ~{p_len} (Actual: {actual_prompt_len}) tokens ---")
        print(llada_generation_df[llada_generation_df['prompt_length'] == actual_prompt_len].to_string(index=False))

    # --- Analysis 3: End-to-End Latency Comparison ---
    print("\n\n" + "-"*80)
    print("\n[ Analysis 3: End-to-End Latency Comparison ]")
    
    # Prepare results for comparison
    llada_e2e_df = llada_generation_df[['prompt_length', 'gen_length', 'total_runtime_ms']].rename(
        columns={'total_runtime_ms': 'LLaDA_total_ms'}
    )
    llama_e2e_df_renamed = llama_e2e_df.rename(columns={'total_runtime_ms': 'Llama-3_total_ms'})

    # Compare for all LLaDA prompt lengths
    for p_len in LLADA_PROMPT_LENGTHS_FOR_TEST:
        print(f"\n--- Comparison for Prompt Length: ~{p_len} tokens ---")

        # Get Llama results (empty if length exceeds Llama's limit)
        llama_subset = llama_e2e_df_renamed[llama_e2e_df_renamed['prompt_length'] == p_len]
        
        # Get LLaDA results
        try:
            range_width = max(30, p_len // 100)  # Wider range for longer prompts
            actual_llada_p_len = llada_e2e_df[llada_e2e_df['prompt_length'].isin(range(p_len, p_len + range_width))]['prompt_length'].iloc[0]
            llada_subset = llada_e2e_df[llada_e2e_df['prompt_length'] == actual_llada_p_len]
        except IndexError:
            # LLaDA results not found (theoretically shouldn't happen)
            llada_subset = pd.DataFrame()

        # Merge results using outer join to show all available data
        if not llama_subset.empty and not llada_subset.empty:
            combined_df = pd.merge(
                llama_subset,
                llada_subset,
                on='gen_length',
                how='outer'
            )
        elif not llama_subset.empty:
            combined_df = llama_subset
        elif not llada_subset.empty:
            combined_df = llada_subset
        else:
            print("No results found for this prompt length.")
            continue
        
        # Organize columns for display
        final_cols = ['prompt_length_x', 'prompt_length_y', 'gen_length', 'prefill_ms', 'decode_ms', 'Llama-3_total_ms', 'LLaDA_total_ms']
        # Filter only existing columns for display
        display_cols = [col for col in final_cols if col in combined_df.columns]
        print(combined_df[display_cols].to_string(index=False))
            
    print("\n" + "="*80)
    print("Profiling complete.")


if __name__ == '__main__':
    main()
