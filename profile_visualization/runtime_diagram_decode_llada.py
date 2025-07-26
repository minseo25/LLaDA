import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log.txt for avg_step_ms analysis
data_128 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'avg_step_ms': [42.529821, 59.459470, 74.052732, 92.291102, 148.655045, 218.590009, 420.473208]
}

data_1024 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'avg_step_ms': [210.071936, 211.916640, 218.750026, 271.158786, 296.338374, 403.941334, 594.477606]
}

data_2048 = {
    'gen_length': [32, 64, 128, 256, 512, 1024, 2048],
    'avg_step_ms': [404.183313, 404.172391, 420.519687, 435.170732, 478.325511, 594.368760, 796.717833]
}

# Create DataFrames
df_128 = pd.DataFrame(data_128)
df_1024 = pd.DataFrame(data_1024)
df_2048 = pd.DataFrame(data_2048)

# Function to create individual graph
def create_avg_step_graph(df, prompt_length, filename):
    plt.figure(figsize=(10, 4))
    plt.style.use('default')
    
    # Display all data points as black dots
    plt.scatter(df['gen_length'], df['avg_step_ms'], 
               s=100, color='black', alpha=0.8, zorder=3, label='Actual Data')
    
    # Add trend line
    z = np.polyfit(df['gen_length'], df['avg_step_ms'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['gen_length'].min(), df['gen_length'].max(), 100)
    plt.plot(x_trend, p(x_trend), color='red', linewidth=2, alpha=0.7, 
             linestyle='--', label='Trend')
    
    # Graph styling
    plt.xlabel('Generation Length (tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Step Time (ms)', fontsize=14, fontweight='bold')
    plt.title(f'Llada-8B-Instruct Average Step Time (Prompt Length: {prompt_length} tokens)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Set axis ranges
    plt.xlim(0, df['gen_length'].max() + 100)
    plt.ylim(0, 850)
    
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
    print(f"Average step time range: {df['avg_step_ms'].min():.2f} - {df['avg_step_ms'].max():.2f} ms")
    print(f"Mean average step time: {df['avg_step_ms'].mean():.2f} ms")
    print(f"Standard deviation: {df['avg_step_ms'].std():.2f} ms")
    print(f"Linear trend equation: y = {z[0]:.6f}x + {z[1]:.6f}")

# Create all three graphs
create_avg_step_graph(df_128, 128, 'avg_step_time_prompt_128.png')
create_avg_step_graph(df_1024, 1024, 'avg_step_time_prompt_1024.png')
create_avg_step_graph(df_2048, 2048, 'avg_step_time_prompt_2048.png')

print("\n=== All graphs have been generated successfully! ===")
print("Files created:")
print("- avg_step_time_prompt_128.png")
print("- avg_step_time_prompt_1024.png") 
print("- avg_step_time_prompt_2048.png")
