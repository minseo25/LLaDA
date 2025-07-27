import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log.txt - LLaMA Prefill vs Decode AI Comparison
# Prompt length: 2048 tokens
# Prefill: processes entire prompt at once
# Decode: processes one token at a time

comparison_data = {
    'phase': ['Prefill', 'Decode'],
    'linear_ai': [1110.3567, 0.9996],
    'attention_ai': [1529.7000, 3.9922],
    'aggregate_ai': [708.9658, 1.0519]
}

# Create DataFrame
df = pd.DataFrame(comparison_data)

# Colors for the bars
colors = ['#FFB3BA', '#BAE1FF', '#B8E6B8']  # Pastel pink, blue, and green

# Create the figure
plt.figure(figsize=(10, 8))

# Set up the bar positions
x_positions = np.arange(len(df['phase'])) + 0.5  # Move groups to center
width = 0.125

# Scale up decode values for better visibility (10x)
scaled_data = df.copy()
scaled_data.loc[scaled_data['phase'] == 'Decode', 'linear_ai'] *= 10
scaled_data.loc[scaled_data['phase'] == 'Decode', 'attention_ai'] *= 10
scaled_data.loc[scaled_data['phase'] == 'Decode', 'aggregate_ai'] *= 10

# Create grouped bar chart
bars1 = plt.bar(x_positions - width, scaled_data['linear_ai'], width, 
               color=colors[0], alpha=0.8, label='Linear AI')
bars2 = plt.bar(x_positions, scaled_data['attention_ai'], width, 
               color=colors[1], alpha=0.8, label='Attention AI')
bars3 = plt.bar(x_positions + width, scaled_data['aggregate_ai'], width, 
               color=colors[2], alpha=0.8, label='Aggregate AI')

# Customize the plot
plt.xlabel('Phase (Prompt: 2048 tokens)', fontsize=14, fontweight='bold')
plt.ylabel('Arithmetic Intensity (AI)', fontsize=14, fontweight='bold')
plt.title('LLaMA-3-8B-Instruct: Prefill vs Decode Arithmetic Intensity Comparison', 
          fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks
plt.xticks(x_positions, df['phase'], fontsize=12)

# Set y-axis limit with some padding
max_ai = max(df['linear_ai'].max(), df['attention_ai'].max(), df['aggregate_ai'].max())
plt.ylim(0, max_ai * 1.1)

# Add legend
plt.legend(fontsize=12, loc='upper center')

# Add grid
plt.grid(True, alpha=0.2, linestyle='--', axis='y')

# Add value labels on bars
for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
    # Linear AI value - show original value, not scaled
    height1 = bar1.get_height()
    original_value1 = df.iloc[i]['linear_ai']
    plt.text(bar1.get_x() + bar1.get_width()/2., height1 + max_ai * 0.01,
            f'{original_value1:.1f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')
    
    # Attention AI value - show original value, not scaled
    height2 = bar2.get_height()
    original_value2 = df.iloc[i]['attention_ai']
    plt.text(bar2.get_x() + bar2.get_width()/2., height2 + max_ai * 0.01,
            f'{original_value2:.1f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')
    
    # Aggregate AI value - show original value, not scaled
    height3 = bar3.get_height()
    original_value3 = df.iloc[i]['aggregate_ai']
    plt.text(bar3.get_x() + bar3.get_width()/2., height3 + max_ai * 0.01,
            f'{original_value3:.1f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the graph
plt.savefig('llama_prefill_vs_decode_ai_comparison.png', dpi=300, bbox_inches='tight')

# Display the graph
plt.show()

# Create additional analysis table
print("=== LLaMA Prefill vs Decode Arithmetic Intensity Analysis ===")
print(f"{'Phase':<10} {'Linear AI':<12} {'Attention AI':<15} {'Aggregate AI':<15}")
print("-" * 55)

for i, row in df.iterrows():
    phase = row['phase']
    linear_ai = row['linear_ai']
    attention_ai = row['attention_ai']
    aggregate_ai = row['aggregate_ai']
    
    print(f"{phase:<10} {linear_ai:<12.1f} {attention_ai:<15.1f} {aggregate_ai:<15.1f}")

print("\n=== Key Observations ===")
print("1. Prefill phase has much higher AI values across all components")
print("2. Decode phase has very low AI values (close to 1)")
print("3. Attention AI is highest in prefill phase")
print("4. This shows why prefill is computationally intensive while decode is memory-bound")

# Calculate ratios
prefill_linear = df[df['phase'] == 'Prefill']['linear_ai'].iloc[0]
decode_linear = df[df['phase'] == 'Decode']['linear_ai'].iloc[0]
prefill_attention = df[df['phase'] == 'Prefill']['attention_ai'].iloc[0]
decode_attention = df[df['phase'] == 'Decode']['attention_ai'].iloc[0]
prefill_aggregate = df[df['phase'] == 'Prefill']['aggregate_ai'].iloc[0]
decode_aggregate = df[df['phase'] == 'Decode']['aggregate_ai'].iloc[0]

print(f"\n=== AI Ratios (Prefill/Decode) ===")
print(f"Linear AI ratio: {prefill_linear/decode_linear:.0f}x")
print(f"Attention AI ratio: {prefill_attention/decode_attention:.0f}x")
print(f"Aggregate AI ratio: {prefill_aggregate/decode_aggregate:.0f}x")

print(f"\nGenerated file: llama_prefill_vs_decode_ai_comparison.png") 