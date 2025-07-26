import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log.txt - End-to-End Latency (LLaMA-3)
llama_data = {
    'prompt_length': [128, 128, 128, 128, 128, 128, 128,
                      1024, 1024, 1024, 1024, 1024, 1024, 1024,
                      2048, 2048, 2048, 2048, 2048, 2048, 2048,
                      4096, 4096, 4096, 4096, 4096, 4096, 4096],
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048,
                   32, 64, 128, 256, 512, 1024, 2048,
                   32, 64, 128, 256, 512, 1024, 2048,
                   32, 64, 128, 256, 512, 1024, 2048],
    'prefill_ms': [28.922796, 26.906729, 27.586699, 27.134180, 27.970791, 26.950836, 26.942253,
                   51.555872, 51.369905, 51.403284, 51.348209, 51.348686, 51.416874, 51.421642,
                   101.749420, 101.883173, 102.030277, 101.800919, 101.731300, 102.236509, 103.809357,
                   159.367323, 158.352613, 158.875704, 158.218622, 159.202814, 158.588886, 158.289433],
    'decode_ms': [935.874939, 1862.109661, 3762.820005, 7526.015043, 15066.858053, 30098.472834, 60958.012104,
                  1163.157463, 2105.485439, 3985.295773, 7750.007391, 15311.757565, 30755.695820, 63258.757353,
                  1426.805019, 2411.434174, 4388.536453, 8359.458923, 16408.779144, 32877.142429, 67371.876478,
                  2141.304731, 3260.592222, 5495.719433, 9993.764877, 19083.975554, 37690.803289, 7879.493952]
}

# Data from log.txt - LLaDA Total Runtime
llada_data = {
    'prompt_length': [141, 141, 141, 141, 141, 141, 141,
                      1037, 1037, 1037, 1037, 1037, 1037, 1037,
                      2059, 2059, 2059, 2059, 2059, 2059, 2059],
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048,
                   32, 64, 128, 256, 512, 1024, 2048,
                   32, 64, 128, 256, 512, 1024, 2048],
    'total_runtime_ms': [680.477142, 1902.703047, 4739.374876, 11813.261032, 38055.691481, 111918.084383, 430564.564943,
                        3361.150980, 6781.332493, 14000.001669, 34708.324671, 75862.623692, 206817.963123, 608745.068550,
                        6466.933012, 12933.516502, 26913.259983, 55701.853752, 122451.330900, 304316.805124, 815839.061499]
}

# Create DataFrames
llama_df = pd.DataFrame(llama_data)
llada_df = pd.DataFrame(llada_data)

# Prompt lengths to analyze
prompt_lengths = [128, 1024, 2048, 4096]
colors = ['#FFB3BA', '#BAE1FF']  # Pastel pink and pastel blue
llada_color = '#000000'  # Black color for LLaDA

# Gen lengths for x-axis
gen_lengths = [32, 64, 128, 256, 512, 1024, 2048]
x_positions = np.arange(len(gen_lengths))

# We'll calculate y-axis limits individually for each graph

# Create individual graphs for each prompt length
for idx, prompt_len in enumerate(prompt_lengths):
    # Create new figure for each prompt length
    plt.figure(figsize=(12, 12))
    
    # Filter LLaMA-3 data for current prompt length
    llama_subset = llama_df[llama_df['prompt_length'] == prompt_len]
    
    # Sort by gen_length to ensure correct order
    llama_subset = llama_subset.sort_values('gen_length')
    
    # Set minimum height for prefill bars
    min_height = 127*4
    prefill_values = np.maximum(llama_subset['prefill_ms'], min_height)
    
    # Create stacked bar chart for LLaMA-3
    bars1 = plt.bar(x_positions, prefill_values, 
                   color=colors[0], alpha=0.8, label='LLaMA-3 Prefill', width=0.6)
    bars2 = plt.bar(x_positions, llama_subset['decode_ms'], 
                   bottom=prefill_values, 
                   color=colors[1], alpha=0.8, label='LLaMA-3 Decode', width=0.6)
    
    # Add LLaDA line plot if data exists
    if prompt_len != 4096:  # LLaDA doesn't have 4096 data
        # Find approximate LLaDA data for this prompt length
        if prompt_len == 128:
            llada_prompt_len = 141
        elif prompt_len == 1024:
            llada_prompt_len = 1037
        elif prompt_len == 2048:
            llada_prompt_len = 2059
        else:
            llada_prompt_len = prompt_len
            
        llada_subset = llada_df[llada_df['prompt_length'] == llada_prompt_len]
        if not llada_subset.empty:
            llada_subset = llada_subset.sort_values('gen_length')
            plt.plot(x_positions, llada_subset['total_runtime_ms'], 
                    marker='o', linewidth=1.5, markersize=4, color=llada_color, 
                    label=f'LLaDA Total Runtime')
    
    # Customize the plot
    plt.xlabel('Generation Length (tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (ms)', fontsize=14, fontweight='bold')
    plt.title(f'LLaMA-3-8B-Instruct vs LLaDA-8B-Instruct (Prompt: {prompt_len} tokens)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    plt.xticks(x_positions, [f'{x}' if x < 1000 else f'{x//1000}K' for x in gen_lengths])
    
    # Calculate y-axis limit for this specific graph
    max_value = 0
    llama_total = llama_subset['prefill_ms'] + llama_subset['decode_ms']
    max_value = max(max_value, llama_total.max())
    
    # Add LLaDA data if available
    if prompt_len != 4096:  # LLaDA doesn't have 4096 data
        # Find approximate LLaDA data for this prompt length
        if prompt_len == 128:
            llada_prompt_len = 141
        elif prompt_len == 1024:
            llada_prompt_len = 1037
        elif prompt_len == 2048:
            llada_prompt_len = 2059
        else:
            llada_prompt_len = prompt_len
            
        llada_subset = llada_df[llada_df['prompt_length'] == llada_prompt_len]
        if not llada_subset.empty:
            max_value = max(max_value, llada_subset['total_runtime_ms'].max())
    
    # Set y-axis limit with some padding, but cap at 200000ms
    y_limit = min(max_value * 1.1, 200000)
    plt.ylim(0, y_limit)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Prefill value
        height1 = bar1.get_height()
        if height1 > 0:
            plt.text(bar1.get_x() + bar1.get_width()/2., height1/2.,
                    f'{llama_subset.iloc[i]["prefill_ms"]:.0f}', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Decode value
        height2 = bar2.get_height()
        if height2 > 0:
            plt.text(bar2.get_x() + bar2.get_width()/2., bar1.get_height() + height2/2.,
                    f'{height2:.0f}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save individual graph
    plt.savefig(f'llama_3_vs_llada_prompt_{prompt_len}.png', dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

# Also create the combined 2x2 subplot version
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('LLaMA-3-8B-Instruct vs LLaDA-8B-Instruct Comparison', fontsize=18, fontweight='bold', y=0.95)

for idx, prompt_len in enumerate(prompt_lengths):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Filter LLaMA-3 data for current prompt length
    llama_subset = llama_df[llama_df['prompt_length'] == prompt_len]
    
    # Sort by gen_length to ensure correct order
    llama_subset = llama_subset.sort_values('gen_length')
    
    # Set minimum height for prefill bars
    min_height = 1270
    prefill_values = np.maximum(llama_subset['prefill_ms'], min_height)
    
    # Create stacked bar chart for LLaMA-3
    bars1 = ax.bar(x_positions, prefill_values, 
                   color=colors[0], alpha=0.8, label='LLaMA-3 Prefill', width=0.6)
    bars2 = ax.bar(x_positions, llama_subset['decode_ms'], 
                   bottom=prefill_values, 
                   color=colors[1], alpha=0.8, label='LLaMA-3 Decode', width=0.6)
    
    # Add LLaDA line plot if data exists
    if prompt_len != 4096:  # LLaDA doesn't have 4096 data
        # Find approximate LLaDA data for this prompt length
        if prompt_len == 128:
            llada_prompt_len = 141
        elif prompt_len == 1024:
            llada_prompt_len = 1037
        elif prompt_len == 2048:
            llada_prompt_len = 2059
        else:
            llada_prompt_len = prompt_len
            
        llada_subset = llada_df[llada_df['prompt_length'] == llada_prompt_len]
        if not llada_subset.empty:
            llada_subset = llada_subset.sort_values('gen_length')
            ax.plot(x_positions, llada_subset['total_runtime_ms'], 
                   marker='o', linewidth=1.5, markersize=4, color=llada_color, 
                   label=f'LLaDA Total (Prompt: {llada_prompt_len})')
    
    # Customize the plot
    ax.set_xlabel('Generation Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Prompt Length: {prompt_len} tokens', fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{x}' if x < 1000 else f'{x//1000}K' for x in gen_lengths])
    
    # Calculate y-axis limit for this specific subplot
    max_value = 0
    llama_total = llama_subset['prefill_ms'] + llama_subset['decode_ms']
    max_value = max(max_value, llama_total.max())
    
    # Add LLaDA data if available
    if prompt_len != 4096:  # LLaDA doesn't have 4096 data
        # Find approximate LLaDA data for this prompt length
        if prompt_len == 128:
            llada_prompt_len = 141
        elif prompt_len == 1024:
            llada_prompt_len = 1037
        elif prompt_len == 2048:
            llada_prompt_len = 2059
        else:
            llada_prompt_len = prompt_len
            
        llada_subset = llada_df[llada_df['prompt_length'] == llada_prompt_len]
        if not llada_subset.empty:
            max_value = max(max_value, llada_subset['total_runtime_ms'].max())
    
    # Set y-axis limit with some padding, but cap at 200000ms
    y_limit = min(max_value * 1.1, 200000)
    ax.set_ylim(0, y_limit)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Prefill value
        height1 = bar1.get_height()
        if height1 > 0:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2.,
                   f'{llama_subset.iloc[i]["prefill_ms"]:.0f}', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Decode value
        height2 = bar2.get_height()
        if height2 > 0:
            ax.text(bar2.get_x() + bar2.get_width()/2., bar1.get_height() + height2/2.,
                   f'{height2:.0f}', ha='center', va='center', fontsize=8, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save combined graph
plt.savefig('llama_3_vs_llada_comparison_combined.png', dpi=300, bbox_inches='tight')

# Display combined graph
plt.show()

# Create additional graph for prompt length 128 with x-axis up to 128
plt.figure(figsize=(10, 6))

# Filter LLaMA-3 data for prompt length 128
llama_subset_128 = llama_df[llama_df['prompt_length'] == 128]
llama_subset_128 = llama_subset_128.sort_values('gen_length')

# Filter data up to gen_length 128
llama_subset_128_filtered = llama_subset_128[llama_subset_128['gen_length'] <= 128]
x_positions_128 = np.arange(len(llama_subset_128_filtered))

# Set minimum height for prefill bars
min_height = 127*4
prefill_values = np.maximum(llama_subset_128_filtered['prefill_ms'], min_height)

# Create stacked bar chart for LLaMA-3
bars1 = plt.bar(x_positions_128, prefill_values, 
               color=colors[0], alpha=0.8, label='LLaMA-3 Prefill', width=0.6)
bars2 = plt.bar(x_positions_128, llama_subset_128_filtered['decode_ms'], 
               bottom=prefill_values, 
               color=colors[1], alpha=0.8, label='LLaMA-3 Decode', width=0.6)

# Add LLaDA line plot
llada_subset_128 = llada_df[llada_df['prompt_length'] == 141]
llada_subset_128_filtered = llada_subset_128[llada_subset_128['gen_length'] <= 128]
llada_subset_128_filtered = llada_subset_128_filtered.sort_values('gen_length')

if not llada_subset_128_filtered.empty:
    plt.plot(x_positions_128, llada_subset_128_filtered['total_runtime_ms'], 
            marker='o', linewidth=1.5, markersize=4, color=llada_color, 
            label='LLaDA Total Runtime')

# Customize the plot
plt.xlabel('Generation Length (tokens)', fontsize=14, fontweight='bold')
plt.ylabel('Runtime (ms)', fontsize=14, fontweight='bold')
plt.title('LLaMA-3-8B-Instruct vs LLaDA-8B-Instruct (Prompt: 128 tokens, Gen: ≤128)', 
          fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks
plt.xticks(x_positions_128, [f'{x}' for x in llama_subset_128_filtered['gen_length']])

# Calculate y-axis limit
max_value = 0
llama_total = llama_subset_128_filtered['prefill_ms'] + llama_subset_128_filtered['decode_ms']
max_value = max(max_value, llama_total.max())

if not llada_subset_128_filtered.empty:
    max_value = max(max_value, llada_subset_128_filtered['total_runtime_ms'].max())

# Set y-axis limit with some padding, but cap at 200000ms
y_limit = min(max_value * 1.1, 200000)
plt.ylim(0, y_limit)

# Add legend
plt.legend(fontsize=12)

# Add grid
plt.grid(True, alpha=0.2, linestyle='--', axis='y')

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    # Prefill value
    height1 = bar1.get_height()
    if height1 > 0:
        plt.text(bar1.get_x() + bar1.get_width()/2., height1/2.,
                f'{llama_subset_128_filtered.iloc[i]["prefill_ms"]:.0f}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Decode value
    height2 = bar2.get_height()
    if height2 > 0:
        plt.text(bar2.get_x() + bar2.get_width()/2., bar1.get_height() + height2/2.,
                f'{height2:.0f}', ha='center', va='center', fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the additional graph
plt.savefig('llama_3_vs_llada_prompt_128_gen_128.png', dpi=300, bbox_inches='tight')
plt.close()

# Print analysis information
print("=== LLaMA-3 vs LLaDA Comparison Analysis ===")
for prompt_len in prompt_lengths:
    llama_subset = llama_df[llama_df['prompt_length'] == prompt_len]
    print(f"\nPrompt Length: {prompt_len} tokens")
    print(f"LLaMA-3 - Average prefill time: {llama_subset['prefill_ms'].mean():.2f} ms")
    print(f"LLaMA-3 - Average decode time: {llama_subset['decode_ms'].mean():.2f} ms")
    print(f"LLaMA-3 - Average total time: {(llama_subset['prefill_ms'] + llama_subset['decode_ms']).mean():.2f} ms")
    
    # LLaDA data if available
    if prompt_len != 4096:
        if prompt_len == 128:
            llada_prompt_len = 141
        elif prompt_len == 1024:
            llada_prompt_len = 1037
        elif prompt_len == 2048:
            llada_prompt_len = 2059
        else:
            llada_prompt_len = prompt_len
            
        llada_subset = llada_df[llada_df['prompt_length'] == llada_prompt_len]
        if not llada_subset.empty:
            print(f"LLaDA - Average total time: {llada_subset['total_runtime_ms'].mean():.2f} ms")
            print(f"LLaDA - Prompt length: {llada_prompt_len} tokens")

print(f"\nGenerated files:")
for prompt_len in prompt_lengths:
    print(f"- llama_3_vs_llada_prompt_{prompt_len}.png")
print("- llama_3_vs_llada_comparison_combined.png")
print("- llama_3_vs_llada_prompt_128_gen_128.png")
print(f"\nEach graph uses its own y-axis range based on the data.")
