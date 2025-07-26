import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log2.txt for decode cost analysis
data_128 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'runtime_ms': [28.054237, 27.939081, 28.666735, 27.821302, 28.104067, 28.307199, 28.147697]
}

data_1024 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'runtime_ms': [28.247356, 28.220654, 28.210163, 28.557301, 28.426170, 28.251648, 28.150558]
}

data_2048 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'runtime_ms': [28.117657, 28.162241, 28.085709, 28.071404, 28.403044, 28.404713, 28.199673]
}

data_4096 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'runtime_ms': [28.280020, 28.125525, 28.196335, 28.233290, 28.179646, 28.205395, 28.674364]
}

# Create DataFrames
df_128 = pd.DataFrame(data_128)
df_1024 = pd.DataFrame(data_1024)
df_2048 = pd.DataFrame(data_2048)
df_4096 = pd.DataFrame(data_4096)

# Function to create individual graph
def create_decode_graph(df, prompt_length, filename):
    plt.figure(figsize=(10, 4))
    plt.style.use('default')
    
    # Display all data points as black dots
    plt.scatter(df['gen_length'], df['runtime_ms'], 
               s=100, color='black', alpha=0.8, zorder=3, label='Actual Data')
    
    # Add trend line
    z = np.polyfit(df['gen_length'], df['runtime_ms'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['gen_length'].min(), df['gen_length'].max(), 100)
    plt.plot(x_trend, p(x_trend), color='red', linewidth=2, alpha=0.7, 
             linestyle='--', label='Linear Trend')
    
    # Graph styling
    plt.xlabel('Generation Length (tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (ms)', fontsize=14, fontweight='bold')
    plt.title(f'Llama-3-8B-Instruct Decode Cost (Prompt Length: {prompt_length} tokens)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Set axis ranges
    plt.xlim(0, df['gen_length'].max() + 100)
    plt.ylim(0, 40)
    
    # Set x-axis ticks
    plt.xticks([0, 256, 512, 1024, 2048])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save graph
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis information
    print(f"\n=== Analysis for Prompt Length: {prompt_length} tokens ===")
    print(f"Total data points: {len(df)}")
    print(f"Generation length range: {df['gen_length'].min()} - {df['gen_length'].max()} tokens")
    print(f"Runtime range: {df['runtime_ms'].min():.2f} - {df['runtime_ms'].max():.2f} ms")
    print(f"Average runtime: {df['runtime_ms'].mean():.2f} ms")
    print(f"Standard deviation: {df['runtime_ms'].std():.2f} ms")
    print(f"Linear trend equation: y = {z[0]:.6f}x + {z[1]:.6f}")

# Create all four graphs
create_decode_graph(df_128, 128, 'decode_cost_prompt_128.png')
create_decode_graph(df_1024, 1024, 'decode_cost_prompt_1024.png')
create_decode_graph(df_2048, 2048, 'decode_cost_prompt_2048.png')
create_decode_graph(df_4096, 4096, 'decode_cost_prompt_4096.png')

print("\n=== All graphs have been generated successfully! ===")
print("Files created:")
print("- decode_cost_prompt_128.png")
print("- decode_cost_prompt_1024.png") 
print("- decode_cost_prompt_2048.png")
print("- decode_cost_prompt_4096.png")
