import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
import re
import json

def run_llama_analysis(sequence_length, input_prompt_length=0):
    """Run LLaMA-3 analysis and parse results."""
    cmd = [
        'python', 'run_llama_roofline_model.py',
        '--sequence_length', str(sequence_length),
        '--input_prompt_length', str(input_prompt_length)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # parse results
    total_runtime_match = re.search(r'Total Runtime: ([\d.]+) s', output)
    total_flops_match = re.search(r'Total FLOPs: ([\d.e+-]+)', output)
    performance_match = re.search(r'Performance: ([\d.]+) GFLOP/s', output)
    agg_ai_match = re.search(r'Aggregate AI: ([\d.]+)', output)
    
    if all([total_runtime_match, total_flops_match, performance_match, agg_ai_match]):
        return {
            'runtime': float(total_runtime_match.group(1)),
            'flops': float(total_flops_match.group(1)),
            'gflops': float(performance_match.group(1)),
            'ai': float(agg_ai_match.group(1))
        }
    else:
        print(f"LLaMA analysis failed: {output}")
        return None

def run_llada_analysis(sequence_length, block_length=512):
    """Run LLaDA analysis and parse results."""
    cmd = [
        'python', 'run_llada_roofline_model.py',
        '--sequence_length', str(sequence_length),
        '--block_length', str(block_length),
        '--steps', str(sequence_length//2)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # parse results
    total_runtime_match = re.search(r'Total Estimated Runtime: ([\d.]+) ms', output)
    total_flops_match = re.search(r'Total FLOPs: ([\d.e+-]+)', output)
    performance_match = re.search(r'Average Performance: ([\d.]+) GFLOP/s', output)
    agg_ai_match = re.search(r'Final Aggregate Arithmetic Intensity \(AI\): ([\d.]+)', output)
    
    if all([total_runtime_match, total_flops_match, performance_match, agg_ai_match]):
        return {
            'runtime': float(total_runtime_match.group(1)) / 1000,  # Convert ms to s
            'flops': float(total_flops_match.group(1)),
            'gflops': float(performance_match.group(1)),
            'ai': float(agg_ai_match.group(1))
        }
    else:
        print(f"LLaDA analysis failed: {output}")
        return None

def plot_roofline_analysis():
    """Visualize A6000 Roofline model and performance data."""
    
    # --- 1. A6000 Hardware Specifications ---
    # NVIDIA RTX A6000 Specs (from gpu_roofline_model.py)
    # Peak FP16 Tensor Core Performance: 84 SMs * 4 TCs/SM * 256 MACs/cycle/TC * 1.41 GHz = 121.3 TFLOPs
    # But actually 2x (FMA operations) = 242.6 TFLOPs
    peak_tflops = 242.6
    # Memory Bandwidth: 768 GB/s (A6000 specs)
    memory_bw_gbs = 768
    
    print("=== A6000 Roofline Model Analysis ===")
    print(f"Peak Compute: {peak_tflops:.1f} TFLOP/s")
    print(f"Memory Bandwidth: {memory_bw_gbs} GB/s")
    print(f"Ridge Point AI: {peak_tflops * 1000 / memory_bw_gbs:.1f} FLOPs/Byte")
    print()
    
    # --- 2. Performance Data Collection ---
    print("=== Collecting Performance Data... ===")
    
    # LLaMA-3 Analysis
    print("LLaMA-3 Prefill (2048 tokens)...")
    llama_prefill = run_llama_analysis(sequence_length=0, input_prompt_length=2048)
    
    print("LLaMA-3 Decode (2048 tokens)...")
    llama_decode = run_llama_analysis(sequence_length=2048, input_prompt_length=0)
    
    # LLaDA Non-SA Analysis (various sequence lengths)
    llada_non_sa = {}
    prompt_length = 16
    for seq_len in [32, 128, 1024, 4096, 32768, 131072]:
        gen_length = seq_len - prompt_length
        print(f"LLaDA Non-SA (prompt={prompt_length}, gen={gen_length}, total={seq_len} tokens)...")
        result = run_llada_analysis(sequence_length=seq_len, block_length=seq_len)
        if result:
            llada_non_sa[str(seq_len)] = result
    
    llada_sa = {}
    for block_len in [32, 64, 128, 256, 512, 1024, 2048]:
        print(f"LLaDA SA (block_length={block_len})...")
        result = run_llada_analysis(sequence_length=2048, block_length=block_len)
        if result:
            llada_sa[str(block_len)] = result
    
    print("=== Data Collection Complete ===")
    print()
    
    # --- 3. Roofline Chart Generation ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Font setting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Draw Roofline
    ai_range = np.logspace(-2, 4, 100)
    ridge_point = peak_tflops * 1000 / memory_bw_gbs
    performance_mem = ai_range * memory_bw_gbs
    performance_compute = np.full_like(ai_range, peak_tflops * 1000)
    roofline = np.minimum(performance_mem, performance_compute)
    
    ax.plot(ai_range, roofline, color='black', lw=3, label='A6000 Roofline (FP16)')
    ax.axhline(y=peak_tflops*1000, color='darkgray', linestyle='--', alpha=0.7, 
               label=f'Peak Compute ({peak_tflops:.0f} TFLOP/s)')
    ax.axvline(x=ridge_point, color='darkgray', linestyle='--', alpha=0.7)
    
    # Data point plotting
    colors = {
        'llama_prefill': '#FFB3BA',  # Pastel pink
        'llama_decode': '#BAE1FF',   # Pastel blue
        'llada_non_sa': '#B8E6B8',   # Pastel green
        'llada_sa': '#DDA0DD'        # Plum purple
    }
    
    # LLaMA-3 points
    if llama_prefill:
        ax.plot(llama_prefill['ai'], llama_prefill['gflops'], 'o', ms=12, 
                color=colors['llama_prefill'], label='LLaMA-3 Prefill (Seq=2K)', 
                markerfacecolor=colors['llama_prefill'], markeredgewidth=2, alpha=0.9)
        ax.annotate("LLaMA prefill", 
                   (llama_prefill['ai'], llama_prefill['gflops']), 
                   xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    if llama_decode:
        ax.plot(llama_decode['ai'], llama_decode['gflops'], 's', ms=12,
                color=colors['llama_decode'], label='LLaMA-3 Decode (Seq=2K)',
                markerfacecolor=colors['llama_decode'], markeredgewidth=2, alpha=0.9)
        ax.annotate("LLaMA decode", 
                   (llama_decode['ai'], llama_decode['gflops']), 
                   xytext=(-10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # LLaDA Non-SA points
    if llada_non_sa:
        non_sa_ais = [v['ai'] for v in llada_non_sa.values()]
        non_sa_gflops = [v['gflops'] for v in llada_non_sa.values()]
        ax.plot(non_sa_ais, non_sa_gflops, 'X', ms=10, color=colors['llada_non_sa'], 
                label='LLaDA Non-SA', markerfacecolor=colors['llada_non_sa'], markeredgewidth=2, alpha=0.9)
        
        # Add labels to each point
        for i, (seq_len, data) in enumerate(llada_non_sa.items()):
            # Convert sequence length to readable format
            if seq_len == "32":
                tag = "Seq32"
            elif seq_len == "128":
                tag = "Seq128"
            elif seq_len == "1024":
                tag = "Seq1K"
            elif seq_len == "4096":
                tag = "Seq4K"
            elif seq_len == "32768":
                tag = "Seq32K"
            elif seq_len == "131072":
                tag = "Seq128K"
            else:
                tag = f"Seq{seq_len}"
            
            # Check if this is likely the LLaDA Prefill point (lowest AI)
            if data['ai'] < 100:  # Assuming LLaDA Prefill has very low AI
                ax.annotate(tag, 
                           (data['ai'], data['gflops']), 
                           xytext=(-10, -15), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            else:
                ax.annotate(tag, 
                           (data['ai'], data['gflops']), 
                           xytext=(-10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    if llada_sa:
        sa_ais = [v['ai'] for v in llada_sa.values()]
        sa_gflops = [v['gflops'] for v in llada_sa.values()]
        ax.plot(sa_ais, sa_gflops, 'D', ms=8, color=colors['llada_sa'], 
                label='LLaDA SA (Seq=2K)', markerfacecolor=colors['llada_sa'], markeredgewidth=2, alpha=0.8)

    # Chart settings
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (AI) [FLOPs/Byte]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance [GFLOP/s]', fontsize=14, fontweight='bold')
    ax.set_title('A6000 Roofline Model\n(LLaMA-3 8B Instruct vs. LLaDA 8B Instruct)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--', axis='both')
    
    # Axis range settings
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(10, 600000)  # Y-axis from 10^1 to higher range with more top margin
    
    plt.tight_layout()
    
    # Save and display results
    plt.savefig('roofline_analysis.png', dpi=300, bbox_inches='tight')
    print("Chart saved as 'roofline_analysis.png'.")
    plt.show()
    
    # --- 4. Results Summary Output ---
    print("\n=== Performance Analysis Summary ===")
    print(f"{'Model':<15} {'AI':<10} {'Performance':<15} {'Bound':<10}")
    print("-" * 50)
    
    if llama_prefill:
        bound = "Compute" if llama_prefill['ai'] > ridge_point else "Memory"
        print(f"{'LLaMA-3 Prefill':<15} {llama_prefill['ai']:<10.0f} {llama_prefill['gflops']:<15.0f} {bound:<10}")
    
    if llama_decode:
        bound = "Compute" if llama_decode['ai'] > ridge_point else "Memory"
        print(f"{'LLaMA-3 Decode':<15} {llama_decode['ai']:<10.1f} {llama_decode['gflops']:<15.0f} {bound:<10}")
    
    for seq_len, data in llada_non_sa.items():
        bound = "Compute" if data['ai'] > ridge_point else "Memory"
        print(f"{'LLaDA Non-SA '+seq_len+'t':<15} {data['ai']:<10.0f} {data['gflops']:<15.0f} {bound:<10}")
    
    for block_len, data in llada_sa.items():
        bound = "Compute" if data['ai'] > ridge_point else "Memory"
        print(f"{'LLaDA SA B'+block_len:<15} {data['ai']:<10.0f} {data['gflops']:<15.0f} {bound:<10}")

if __name__ == '__main__':
    plot_roofline_analysis() 