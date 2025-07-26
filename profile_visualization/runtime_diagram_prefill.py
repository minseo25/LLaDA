import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm

# Font setting
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from log.txt
data = {
    'prompt_length': [32, 64, 128, 256, 512, 1024, 2048, 4096],
    'runtime_ms': [27.506065, 27.946615, 28.027058, 40.930891, 81.018829, 173.521805, 331.910753, 673.928452]
}

# Create DataFrame
df = pd.DataFrame(data)

# Graph settings
plt.figure(figsize=(12, 8))
plt.style.use('default')

# Display all data points as black dots
plt.scatter(df['prompt_length'], df['runtime_ms'], 
           s=100, color='black', alpha=0.8, zorder=3, label='Actual Data')

# Add quadratic trend line (N^2)
z_quad = np.polyfit(df['prompt_length'], df['runtime_ms'], 2)
p_quad = np.poly1d(z_quad)
x_trend = np.linspace(df['prompt_length'].min(), df['prompt_length'].max(), 100)
plt.plot(x_trend, p_quad(x_trend), color='red', linewidth=2, alpha=0.7, 
         linestyle='--', label='Quadratic Trend (N²)')

# Graph styling
plt.xlabel('Prompt Length (tokens)', fontsize=14, fontweight='bold')
plt.ylabel('Runtime (ms)', fontsize=14, fontweight='bold')
plt.title('LLaMA-3-8B-Instruct Prefill Cost', fontsize=16, fontweight='bold', pad=20)

# Add legend
plt.legend(fontsize=12)

# Add grid
plt.grid(True, alpha=0.2, linestyle='--')

# Set axis ranges
plt.xlim(0, df['prompt_length'].max() + 200)
plt.ylim(0, df['runtime_ms'].max() + 50)

# Set x-axis ticks (key values only)
plt.xticks([0, 128, 256, 512, 1024, 2048, 4096])

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig('prefill_llama_3_analysis.png', dpi=300, bbox_inches='tight')

# Display graph
plt.show()

# Print additional analysis information
print(f"Total data points: {len(df)}")
print(f"Prompt length range: {df['prompt_length'].min()} - {df['prompt_length'].max()} tokens")
print(f"Runtime range: {df['runtime_ms'].min():.2f} - {df['runtime_ms'].max():.2f} ms")
print(f"Correlation coefficient: {df['prompt_length'].corr(df['runtime_ms']):.4f}")
print(f"Quadratic trend equation: y = {z_quad[0]:.6f}x² + {z_quad[1]:.6f}x + {z_quad[2]:.6f}")
