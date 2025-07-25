import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import random
from generate import generate as generate_llada
import numpy as np

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LLAMA_MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLADA_MODEL_ID = 'GSAI-ML/LLaDA-8B-Instruct'


# --- Helper Functions ---
def get_gsm8k_prompts(tokenizer, num_prompts=100):
    """Sample and sort prompts from GSM8K for prefill analysis."""
    print("Loading GSM8K dataset for prefill prompts...")
    dataset = load_dataset("openai/gsm8k", "main", cache_dir='/rscratch/minseokim/hf_dataset', split="train")
    num_prompts = min(num_prompts, len(dataset))
    indices = random.sample(range(len(dataset)), num_prompts)
    sampled = [dataset[i]['question'] for i in indices]
    sampled_sorted = sorted(sampled, key=lambda x: len(tokenizer(x)['input_ids']))
    print(f"Loaded {len(sampled_sorted)} prompts from GSM8K.")
    return sampled_sorted


def get_long_prompts():
    """
    Return various prompts to induce long responses.
    (Token counts are based on Llama-3-8B tokenizer, all 46 tokens)
    """
    return [
        "Provide a detailed explanation of how diffusion models work for image generation. Describe the forward process of adding noise and the reverse process where a network learns to denoise the image, and include key concepts like classifier-free guidance and timesteps.",
        "Provide a comprehensive summary of the key events of World War II, covering both the European and Pacific theaters. Your summary should explain the primary causes, major turning points, and the ultimate resolution of conflicts, including significant battles and strategies.",
        "Write a creative short story about a lone space explorer who crash-lands and discovers a planet inhabited by sentient crystalline beings. Your story must describe their unique society, their method of non-verbal communication, and the central conflict.",
        "Explain blockchain technology from the ground up, detailing its core components. Describe what blocks and chains are, how mining works, and the role of a consensus algorithm. Please provide several real-world examples of its use beyond just cryptocurrencies."
    ]


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
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    return time.time()


# --- 1. Prefill Cost Analysis ---
def profile_llama_prefill(model, tokenizer, prompts):
    """Measure the runtime of the Prefill step of Llama-3."""
    print("\n--- Profiling Llama-3 Prefill Cost ---")
    results = []
    for prompt_text in tqdm(prompts, desc="Profiling Prefill"):
        input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
        prompt_length = input_ids.shape[1]
        
        with torch.no_grad():
            _ = model(input_ids) # Warm-up

        start_time = synchronize_and_time()
        with torch.no_grad():
            _ = model(input_ids)
        end_time = synchronize_and_time()
        
        duration_ms = (end_time - start_time) * 1000
        results.append({'prompt_length': prompt_length, 'runtime_ms': duration_ms})
    
    return results


# --- 2. Unit Generation Cost Analysis ---
def profile_llama_decode(model, tokenizer, prompts, max_gen_length):
    """Measure Llama-3 decode runtime per token, averaged over multiple prompts."""
    print("\n--- Profiling Llama-3 Decode Cost ---")
    all_results_df = pd.DataFrame()
    prompt_lengths = []
    for prompt_text in tqdm(prompts, desc="Profiling Decode"):
        results = []
        input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
        prompt_length = input_ids.shape[1]
        prompt_lengths.append(prompt_length)
        print(f"Prompt length: {prompt_length}")

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        for i in range(max_gen_length):
            start_time = synchronize_and_time()
            with torch.no_grad():
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            end_time = synchronize_and_time()
            duration_ms = (end_time - start_time) * 1000
            gen_length = i + 1
            results.append({'gen_length': gen_length, 'runtime_ms': duration_ms})
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            if (next_token == tokenizer.eos_token_id).all():
                break
        all_results_df = pd.concat([all_results_df, pd.DataFrame(results)])
    avg_results = all_results_df.groupby('gen_length')['runtime_ms'].mean().reset_index()
    return avg_results, prompt_lengths


def profile_llada_step_cost(model, tokenizer, prompts, gen_lengths):
    """Measure LLaDA average step runtime, averaged over multiple prompts."""
    print("\n--- Profiling LLaDA Step Cost ---")
    all_results_df = pd.DataFrame()
    prompt_lengths = []
    for prompt_text in tqdm(prompts, desc="Profiling LLaDA Steps"):
        results = []
        m = [{"role": "user", "content": prompt_text}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input, return_tensors='pt').input_ids.to(DEVICE)
        prompt_length = input_ids.shape[1]
        prompt_lengths.append(prompt_length)
        print(f"Prompt length: {prompt_length}")

        for gen_len in gen_lengths:
            steps = max(1, gen_len // 2)
            print(f"Steps: {steps} | Gen length: {gen_len}")
            with torch.no_grad():
                _ = generate_llada(model, input_ids, steps=steps, gen_length=gen_len, block_length=32)
            start_time = synchronize_and_time()
            with torch.no_grad():
                _ = generate_llada(model, input_ids, steps=steps, gen_length=gen_len, block_length=32)
            end_time = synchronize_and_time()
            total_duration_ms = (end_time - start_time) * 1000
            avg_step_ms = total_duration_ms / steps if steps > 0 else 0
            results.append({'gen_length': gen_len, 'avg_step_ms': avg_step_ms})
        all_results_df = pd.concat([all_results_df, pd.DataFrame(results)])
    avg_results = all_results_df.groupby('gen_length')['avg_step_ms'].mean().reset_index()
    return avg_results, prompt_lengths


# --- 3. End-to-End Latency Analysis ---
def profile_e2e_latency(model, tokenizer, model_name, prompt_text, gen_lengths):
    """
    Measure the end-to-end latency of the model.
    - Llama-3: measure the total time of Prefill and Decode
    - LLaDA: measure the total time of generation
    """
    print(f"\n--- Profiling End-to-End Latency for {model_name} ---")
    results = []
    
    # Create input_ids based on the model's input format
    if model_name == 'Llama-3':
        input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
        prompt_length = input_ids.shape[1]
    elif model_name == 'LLaDA':
        m = [{"role": "user", "content": prompt_text}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input, return_tensors='pt').input_ids.to(DEVICE)
        prompt_length = input_ids.shape[1]

    for gen_len in tqdm(gen_lengths, desc=f"Profiling E2E ({model_name})"):
        # Warm-up run is only done once for the entire generate function
        with torch.no_grad():
            if model_name == 'Llama-3':
                _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            elif model_name == 'LLaDA':
                _ = generate_llada(model, input_ids, steps=max(1, gen_len // 2), gen_length=gen_len, block_length=32)
        
        # --- Start actual measurement ---
        if model_name == 'Llama-3':
            # 1. Measure Prefill time
            start_prefill = synchronize_and_time()
            with torch.no_grad():
                # model() call to only perform Prefill
                _ = model(input_ids)
            end_prefill = synchronize_and_time()
            prefill_ms = (end_prefill - start_prefill) * 1000

            # 2. Measure total generation time
            start_total = synchronize_and_time()
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            end_total = synchronize_and_time()
            total_ms = (end_total - start_total) * 1000
            
            # 3. Calculate Decode time (total time - Prefill time)
            decode_ms = max(0, total_ms - prefill_ms) # avoid negative values

            results.append({
                'prompt_length': prompt_length,
                'total_seq_len': prompt_length + gen_len,
                'gen_length': gen_len,
                'prefill_ms': prefill_ms,
                'decode_ms': decode_ms,
                'total_runtime_ms': total_ms
            })
        
        elif model_name == 'LLaDA':
            # LLaDA only measures the total time
            start_total = synchronize_and_time()
            with torch.no_grad():
                _ = generate_llada(model, input_ids, steps=max(1, gen_len // 2), gen_length=gen_len, block_length=32)
            end_total = synchronize_and_time()
            total_ms = (end_total - start_total) * 1000
            
            results.append({
                'prompt_length': prompt_length,
                'total_seq_len': prompt_length + gen_len,
                'gen_length': gen_len,
                'total_runtime_ms': total_ms
            })

    return pd.DataFrame(results)

# --- Main function ---
def main():
    """Main execution function"""
    if DEVICE == 'cpu':
        print("="*80 + "\n!! Warning: Running on CPU is not feasible for 8B models. Exiting. !!\n" + "="*80)
        exit()

    # --- Experiment Parameters ---
    PREFILL_PROMPTS_COUNT = 1000
    UNIT_COST_PROMPTS = get_long_prompts()
    MAX_DECODE_LENGTH = 1024
    GEN_LENGTHS_FOR_TESTS = [32, 64, 128, 256, 512, 1024]

    # 1. Llama-3 analysis
    llama_model, llama_tokenizer = load_model_and_tokenizer(LLAMA_MODEL_ID, AutoModelForCausalLM)
    prefill_prompts = get_gsm8k_prompts(llama_tokenizer, PREFILL_PROMPTS_COUNT)
    llama_prefill_results_df = pd.DataFrame(profile_llama_prefill(llama_model, llama_tokenizer, prefill_prompts))
    llama_decode_results_df, llama_decode_prompt_lengths = profile_llama_decode(llama_model, llama_tokenizer, UNIT_COST_PROMPTS, MAX_DECODE_LENGTH)
    shortest_prompt, median_prompt, longest_prompt = prefill_prompts[0], prefill_prompts[len(prefill_prompts)//2], prefill_prompts[-1]
    llama_e2e_short_df = profile_e2e_latency(llama_model, llama_tokenizer, 'Llama-3', shortest_prompt, GEN_LENGTHS_FOR_TESTS)
    llama_e2e_median_df = profile_e2e_latency(llama_model, llama_tokenizer, 'Llama-3', median_prompt, GEN_LENGTHS_FOR_TESTS)
    llama_e2e_long_df = profile_e2e_latency(llama_model, llama_tokenizer, 'Llama-3', longest_prompt, GEN_LENGTHS_FOR_TESTS)
    del llama_model, llama_tokenizer
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    # 2. LLaDA analysis
    llada_model, llada_tokenizer = load_model_and_tokenizer(LLADA_MODEL_ID, AutoModel)
    llada_step_results_df, llada_step_prompt_lengths = profile_llada_step_cost(llada_model, llada_tokenizer, UNIT_COST_PROMPTS, GEN_LENGTHS_FOR_TESTS)
    llada_e2e_short_df = profile_e2e_latency(llada_model, llada_tokenizer, 'LLaDA', shortest_prompt, GEN_LENGTHS_FOR_TESTS)
    llada_e2e_median_df = profile_e2e_latency(llada_model, llada_tokenizer, 'LLaDA', median_prompt, GEN_LENGTHS_FOR_TESTS)
    llada_e2e_long_df = profile_e2e_latency(llada_model, llada_tokenizer, 'LLaDA', longest_prompt, GEN_LENGTHS_FOR_TESTS)
    del llada_model, llada_tokenizer
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    # 3. Print all results
    pd.set_option('display.width', 150)
    pd.set_option('display.max_rows', 50)
    print("\n\n" + "="*28 + " FINAL ANALYSIS RESULTS " + "="*28)

    # --- Analysis 1: Prefill Cost ---
    print("\n[ Analysis 1: Llama-3 Prefill Cost ]")
    print("Average prefill cost by prompt length (ms):")
    prefill_summary = llama_prefill_results_df.groupby('prompt_length')['runtime_ms'].mean().reset_index()
    print(prefill_summary.to_string(index=False))

    # --- Analysis 2: Unit Generation Cost ---
    print("\n\n" + "-"*80)
    print("\n[ Analysis 2: Unit Generation Cost Comparison ]")
    print("Llama-3's cost per token vs. LLaDA's average cost per step.")
    avg_prompt_len_unit = np.mean([len(llama_tokenizer(p)['input_ids']) for p in UNIT_COST_PROMPTS])
    print(f"(Average prompt length for this test: {avg_prompt_len_unit:.1f} tokens)")
    
    # Merge for a side-by-side comparison
    unit_cost_comparison_df = pd.merge(
        llama_decode_results_df.rename(columns={'runtime_ms': 'Llama-3_per_token_ms'}),
        llada_step_results_df.rename(columns={'avg_step_ms': 'LLaDA_per_step_ms'}),
        on='gen_length',
        how='inner'
    )
    print(unit_cost_comparison_df.to_string(index=False))

    # --- Analysis 3: End-to-End Latency ---
    print("\n\n" + "-"*80)
    print("\n[ Analysis 3: End-to-End Latency Comparison ]")
    print("Total runtime for shortest, median, and longest GSM8K prompts.")

    # Helper function to create combined tables for cleaner output
    def combine_e2e_results(llama_df, llada_df, prompt_name):
        llada_renamed = llada_df[['gen_length', 'total_runtime_ms']].rename(
            columns={'total_runtime_ms': 'LLaDA_total_ms'}
        )
        combined_df = pd.merge(llama_df, llada_renamed, on='gen_length')
        
        combined_df = combined_df.rename(columns={'total_runtime_ms': 'Llama-3_total_ms'})
        final_cols = ['gen_length', 'prefill_ms', 'decode_ms', 'Llama-3_total_ms', 'LLaDA_total_ms']
        
        prompt_len = llama_df['prompt_length'].iloc[0]
        print(f"\n--- {prompt_name} Prompt (length: {prompt_len} tokens) ---")
        print(combined_df[final_cols].to_string(index=False))

    combine_e2e_results(llama_e2e_short_df, llada_e2e_short_df, 'Shortest')
    combine_e2e_results(llama_e2e_median_df, llada_e2e_median_df, 'Median')
    combine_e2e_results(llama_e2e_long_df, llada_e2e_long_df, 'Longest')
    
    print("\n" + "="*80)
    print("Profiling complete.")


if __name__ == '__main__':
    main()
