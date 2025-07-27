from gpu_roofline_model import compute_mmm_runtime_gpu, compute_nonlinear_runtime_gpu

def compute_llada_step_runtime_gpu_roofline_model(
    total_sequence_length=128,
    hidden_dim=4096,
    num_attn_heads=32,
    proj_factor=3.5,
    num_decoder_layers=32,
    batch_size=1
):
    """
    Models the runtime of a single inference step of a non-causal Transformer (like LLaDA's backbone).
    This is analogous to the 'prefill' phase of an ARM.
    """
    gqa_factor = 1 # LLaDA uses Multi-Head Attention
    head_dim = int(hidden_dim / num_attn_heads)

    # --- Calculations for a single layer ---
    # QKV Projections
    A = [hidden_dim, hidden_dim]; B = [hidden_dim, total_sequence_length * batch_size]
    q_time, _, q_flops, q_mops = compute_mmm_runtime_gpu(A, B)
    k_time, _, k_flops, k_mops = compute_mmm_runtime_gpu(A, B)
    v_time, _, v_flops, v_mops = compute_mmm_runtime_gpu(A, B)

    # RoPE
    A = [hidden_dim, total_sequence_length * batch_size]
    ropeq_time, _, ropeq_flops, ropeq_mops = compute_nonlinear_runtime_gpu(A, operation='rope')
    ropek_time, _, ropek_flops, ropek_mops = compute_nonlinear_runtime_gpu(A, operation='rope')

    # Attention (non-causal)
    A = [total_sequence_length, head_dim]; B = [head_dim, total_sequence_length]
    qk_time, _, qk_flops, qk_mops = compute_mmm_runtime_gpu(A, B, nowrite=True, causal=False) # bidirectional!!
    qk_time *= batch_size * num_attn_heads; qk_flops *= batch_size * num_attn_heads; qk_mops *= batch_size * num_attn_heads

    A = [total_sequence_length, total_sequence_length]
    smax_time, _, smax_flops, smax_mops = compute_nonlinear_runtime_gpu(A, operation='softmax', noload=True, nowrite=True, causal=False) # bidirectional!!
    smax_time *= batch_size * num_attn_heads; smax_flops *= batch_size * num_attn_heads; smax_mops *= batch_size * num_attn_heads

    A = [total_sequence_length, total_sequence_length]; B = [total_sequence_length, head_dim]
    score_time, _, score_flops, score_mops = compute_mmm_runtime_gpu(A, B, noloadA=True, causal=False) # bidirectional!!
    score_time *= batch_size * num_attn_heads; score_flops *= batch_size * num_attn_heads; score_mops *= batch_size * num_attn_heads

    # Output Projection
    A = [hidden_dim, hidden_dim]; B = [hidden_dim, total_sequence_length * batch_size]
    o_time, _, o_flops, o_mops = compute_mmm_runtime_gpu(A, B)

    # FFN and LayerNorm
    A = [batch_size * total_sequence_length, hidden_dim]
    ln1_time, _, ln1_flops, ln1_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd1_time, _, eadd1_flops, eadd1_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')
    A = [proj_factor * hidden_dim, hidden_dim]; B = [hidden_dim, total_sequence_length * batch_size]
    ffn1_time, _, ffn1_flops, ffn1_mops = compute_mmm_runtime_gpu(A, B)
    gate_time, _, gate_flops, gate_mops = compute_mmm_runtime_gpu(A, B)
    A = [batch_size * total_sequence_length, proj_factor * hidden_dim]
    sigmoid_time, _, sigmoid_flops, sigmoid_mops = compute_nonlinear_runtime_gpu(A, operation='sigmoid')
    elemmul_time, _, elemmul_flops, elemmul_mops = compute_nonlinear_runtime_gpu(A, operation='elemmul')
    A = [hidden_dim, proj_factor * hidden_dim]; B = [proj_factor * hidden_dim, total_sequence_length * batch_size]
    ffn2_time, _, ffn2_flops, ffn2_mops = compute_mmm_runtime_gpu(A, B)
    A = [total_sequence_length * batch_size, hidden_dim]
    ln2_time, _, ln2_flops, ln2_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd2_time, _, eadd2_flops, eadd2_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')
    lnf_time, _, lnf_flops, lnf_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')

    # --- Aggregate metrics for one full model pass (one step) ---
    total_layer_runtime = (q_time + k_time + v_time + ropeq_time + ropek_time + qk_time + smax_time + score_time + o_time +
                           ln1_time + eadd1_time + ffn1_time + gate_time + sigmoid_time + elemmul_time + ffn2_time + ln2_time + eadd2_time)
    
    total_runtime_one_step = total_layer_runtime * num_decoder_layers + lnf_time

    total_layer_flops = (q_flops + k_flops + v_flops + ropeq_flops + ropek_flops + qk_flops + smax_flops + score_flops + o_flops +
                         ln1_flops + eadd1_flops + ffn1_flops + gate_flops + sigmoid_flops + elemmul_flops + ffn2_flops + ln2_flops + eadd2_flops)
    
    total_flops_one_step = total_layer_flops * num_decoder_layers + lnf_flops

    total_layer_mops = (q_mops + k_mops + v_mops + ropeq_mops + ropek_mops + qk_mops + smax_mops + score_mops + o_mops +
                        ln1_mops + eadd1_mops + ffn1_mops + gate_mops + sigmoid_mops + elemmul_mops + ffn2_mops + ln2_mops + eadd2_mops)

    total_mops_one_step = total_layer_mops * num_decoder_layers + lnf_mops

    return total_runtime_one_step, total_flops_one_step, total_mops_one_step
