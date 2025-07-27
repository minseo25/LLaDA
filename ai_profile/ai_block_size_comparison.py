import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log.txt - LLaDA Block Size Ablation Study
# Fixed prompt length: 128, Fixed sequence length: 512
# Different block sizes: 512, 256, 128, 64, 32

block_size_data = {
    'block_size': [512, 256, 128, 64, 32],
    'aggregate_ai': [400.2126, 345.9858, 315.4992, 299.2796, 290.9062],
    'total_runtime_ms': [11287.7037, 9028.5001, 8231.2754, 7832.6984, 7633.4187],
    'blocks': [1, 2, 4, 8, 16],
    'mode': ['Non-Semi-Autoregressive', 'Semi-Autoregressive', 'Semi-Autoregressive', 'Semi-Autoregressive', 'Semi-Autoregressive']
}

# Create DataFrame
df = pd.DataFrame(block_size_data)

# Colors for the bars
colors = ['#FFB3BA', '#BAE1FF', '#B8E6B8', '#FFE4B5', '#E6B8FF']  # Pastel colors

# Create the figure
plt.figure(figsize=(12, 8))

# Create uniform x positions for better spacing
x_positions = np.arange(len(df['block_size']))
width = 0.6  # Wider bars

# Create bar chart for Aggregate AI
bars = plt.bar(x_positions, df['aggregate_ai'], 
               color=colors, alpha=0.8, width=width)

# Customize the plot
plt.xlabel('Block Size (tokens)', fontsize=14, fontweight='bold')
plt.ylabel('Aggregate Arithmetic Intensity (AI)', fontsize=14, fontweight='bold')
plt.title('LLaDA-8B-Instruct: Block Size vs Arithmetic Intensity\n(Prompt: 128, Generation: 512 tokens)', 
          fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks with block size labels
plt.xticks(x_positions, df['block_size'], fontsize=12)

# Set y-axis limit with some padding
max_ai = df['aggregate_ai'].max()
plt.ylim(0, max_ai * 1.1)

# Add grid
plt.grid(True, alpha=0.2, linestyle='--', axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max_ai * 0.01,
            f'{height:.1f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# Add mode information as text annotations
for i, row in df.iterrows():
    mode_text = "Non-SA" if row['mode'] == 'Non-Semi-Autoregressive' else "SA"
    plt.text(x_positions[i], row['aggregate_ai'] * 0.5, 
            f"{mode_text}\n{row['blocks']} blocks", 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save the graph
plt.savefig('llada_block_size_ai_comparison.png', dpi=300, bbox_inches='tight')

# Display the graph
plt.show()

# Create additional analysis table
print("=== LLaDA Block Size Ablation Study Analysis ===")
print(f"{'Block Size':<12} {'AI':<12} {'Runtime(ms)':<15} {'Blocks':<10} {'Mode':<20}")
print("-" * 75)

for i, row in df.iterrows():
    block_size = row['block_size']
    ai = row['aggregate_ai']
    runtime = row['total_runtime_ms']
    blocks = row['blocks']
    mode = "Non-SA" if row['mode'] == 'Non-Semi-Autoregressive' else "SA"
    
    print(f"{block_size:<12} {ai:<12.1f} {runtime:<15.1f} {blocks:<10} {mode:<20}")

print("\n=== Key Observations ===")
print("1. Smaller block sizes generally lead to lower AI values")
print("2. Block size 512 (Non-SA) has the highest AI: 400.2")
print("3. Block size 32 (SA) has the lowest AI: 290.9")
print("4. AI decreases by ~27% from largest to smallest block size")
print("5. Smaller blocks enable semi-autoregressive generation")

# Calculate performance metrics
ai_decrease = ((df['aggregate_ai'].iloc[0] - df['aggregate_ai'].iloc[-1]) / df['aggregate_ai'].iloc[0]) * 100
runtime_decrease = ((df['total_runtime_ms'].iloc[0] - df['total_runtime_ms'].iloc[-1]) / df['total_runtime_ms'].iloc[0]) * 100

print(f"\n=== Performance Changes (512 → 32 block size) ===")
print(f"AI decrease: {ai_decrease:.1f}%")
print(f"Runtime decrease: {runtime_decrease:.1f}%")

print(f"\nGenerated file: llada_block_size_ai_comparison.png") 