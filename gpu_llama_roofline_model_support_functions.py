# chooper1/PerformanceModeling/tree/aditya-updated
from gpu_roofline_model import compute_mmm_runtime_gpu, compute_nonlinear_runtime_gpu


# prefill
def compute_llm_prompt_runtime_gpu_roofline_model(
    start_sequence_length=0,
    end_sequence_length=128,
    hidden_dim=768,
    num_attn_heads=12,
    gqa_factor=1,
    proj_factor=4,
    num_decoder_layers=12,
    include_classifier=False,
    num_classes=32000,
    batch_size=1
):
    # allow for only part of prompt to be new
    sequence_length = end_sequence_length - start_sequence_length
    q_sequence_length = sequence_length
    kv_sequence_length = end_sequence_length

    # gqa params
    num_kv_heads = int(num_attn_heads / gqa_factor)
    head_dim = int(hidden_dim / num_attn_heads)

    # q projection
    A = [hidden_dim, hidden_dim]
    B = [hidden_dim, sequence_length * batch_size]
    q_time, _, q_flops, q_mops = compute_mmm_runtime_gpu(A, B)

    # kv projections (divide by GQA factor)
    A = [hidden_dim / gqa_factor, hidden_dim]
    B = [hidden_dim, sequence_length * batch_size]
    k_time, _, k_flops, k_mops = compute_mmm_runtime_gpu(A, B)
    v_time, _, v_flops, v_mops = compute_mmm_runtime_gpu(A, B)

    # RoPE
    A = [hidden_dim, sequence_length * batch_size]
    ropeq_time, _, ropeq_flops, ropeq_mops = compute_nonlinear_runtime_gpu(A, operation='rope')
    A = [hidden_dim / gqa_factor, sequence_length * batch_size]
    ropek_time, _, ropek_flops, ropek_mops = compute_nonlinear_runtime_gpu(A, operation='rope')

    # fuse these operations (approximately) to simulate flash attention
    A = [gqa_factor * q_sequence_length, head_dim]
    B = [head_dim, kv_sequence_length]
    qk_time, _, qk_flops, qk_mops = compute_mmm_runtime_gpu(A, B, nowrite=True, causal=True)
    qk_time = qk_time * batch_size * num_kv_heads
    qk_flops = qk_flops * batch_size * num_kv_heads
    qk_mops = qk_mops * batch_size * num_kv_heads

    A = [gqa_factor * q_sequence_length, kv_sequence_length]
    smax_time, _, smax_flops, smax_mops = compute_nonlinear_runtime_gpu(A, operation='softmax', noload=True, nowrite=True, causal=True)
    smax_time = smax_time * batch_size * num_kv_heads
    smax_flops = smax_flops * batch_size * num_kv_heads
    smax_mops = smax_mops * batch_size * num_kv_heads

    A = [gqa_factor * sequence_length, kv_sequence_length]
    B = [kv_sequence_length, head_dim]
    score_time, _, score_flops, score_mops = compute_mmm_runtime_gpu(A, B, noloadA=True, causal=True)
    score_time = score_time * batch_size * num_kv_heads
    score_flops = score_flops * batch_size * num_kv_heads
    score_mops = score_mops * batch_size * num_kv_heads

    # output projection
    A = [hidden_dim, hidden_dim]
    B = [hidden_dim, sequence_length * batch_size]
    o_time, _, o_flops, o_mops = compute_mmm_runtime_gpu(A, B)

    # layernorm + add 1
    A = [batch_size * sequence_length, hidden_dim]
    ln1_time, _, ln1_flops, ln1_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd1_time, _, eadd1_flops, eadd1_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')

    # fc1
    A = [proj_factor * hidden_dim, hidden_dim]
    B = [hidden_dim, sequence_length * batch_size]
    ffn1_time, _, ffn1_flops, ffn1_mops = compute_mmm_runtime_gpu(A, B)

    # gate runtime
    A = [proj_factor * hidden_dim, hidden_dim]
    B = [hidden_dim, sequence_length * batch_size]
    gate_time, _, gate_flops, gate_mops = compute_mmm_runtime_gpu(A, B)

    # sigmoid
    A = [batch_size * sequence_length, proj_factor * hidden_dim]
    sigmoid_time, _, sigmoid_flops, sigmoid_mops = compute_nonlinear_runtime_gpu(A, operation='sigmoid')

    # elemmul
    A = [batch_size * sequence_length, proj_factor * hidden_dim]
    elemmul_time, _, elemmul_flops, elemmul_mops = compute_nonlinear_runtime_gpu(A, operation='elemmul')

    # fc2
    A = [hidden_dim, proj_factor * hidden_dim]
    B = [proj_factor * hidden_dim, sequence_length * batch_size]
    ffn2_time, _, ffn2_flops, ffn2_mops = compute_mmm_runtime_gpu(A, B)

    # layernorm + add 2
    A = [sequence_length * batch_size, hidden_dim]
    ln2_time, _, ln2_flops, ln2_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd2_time, _, eadd2_flops, eadd2_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')

    # final layernorm
    lnf_time, _, lnf_flops, lnf_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')

    # if we want to include vocab layer (important for smaller models)
    if include_classifier:
        A = [num_classes, hidden_dim]
        # modification here - only apply classifier to the final output token
        B = [hidden_dim, batch_size]  # B = [hidden_dim,sequence_length * batch_size]
        emb_time, _, emb_flops, emb_mops = compute_mmm_runtime_gpu(A, B)
    else:
        emb_time, emb_flops, emb_mops = (0, 0, 0)

    # compute total runtime
    ffn_linear_time = (ffn1_time + gate_time + ffn2_time) * num_decoder_layers
    mha_linear_time = (q_time + k_time + v_time + o_time) * num_decoder_layers
    mha_act2act_time = (qk_time + score_time) * num_decoder_layers
    nonlinear_time = (smax_time + ln1_time + ln2_time + eadd1_time + eadd2_time + sigmoid_time + elemmul_time + ropeq_time + ropek_time) * num_decoder_layers + lnf_time
    total_runtime = ffn_linear_time + mha_linear_time + mha_act2act_time + nonlinear_time + emb_time

    # compute ai - flops / mops
    ffn_linear_flops = (ffn1_flops + gate_flops + ffn2_flops) * num_decoder_layers
    mha_linear_flops = (q_flops + k_flops + v_flops + o_flops) * num_decoder_layers
    mha_act2act_flops = (qk_flops + score_flops) * num_decoder_layers
    nonlinear_flops = (smax_flops + ln1_flops + ln2_flops + eadd1_flops + eadd2_flops + sigmoid_flops + elemmul_flops + ropeq_flops + ropek_flops) * num_decoder_layers + lnf_flops
    total_flops = ffn_linear_flops + mha_linear_flops + mha_act2act_flops + nonlinear_flops + emb_flops

    ffn_linear_mops = (ffn1_mops + gate_mops + ffn2_mops) * num_decoder_layers
    mha_linear_mops = (q_mops + k_mops + v_mops + o_mops) * num_decoder_layers
    mha_act2act_mops = (qk_mops + score_mops) * num_decoder_layers
    nonlinear_mops = (smax_mops + ln1_mops + ln2_mops + eadd1_mops + eadd2_mops + sigmoid_mops + elemmul_mops + ropeq_mops + ropek_mops) * num_decoder_layers + lnf_mops
    total_mops = ffn_linear_mops + mha_linear_mops + mha_act2act_mops + nonlinear_mops + emb_mops

    # compute AI for linear and attention ops
    linear_ai = (ffn_linear_flops + mha_linear_flops + emb_flops) / (ffn_linear_mops + mha_linear_mops + emb_mops)
    attention_ai = mha_act2act_flops / mha_act2act_mops

    # aggregate AI
    aggregate_ai = total_flops / total_mops

    return total_runtime, ffn_linear_time, mha_linear_time, mha_act2act_time, nonlinear_time, emb_time, linear_ai, attention_ai, aggregate_ai, total_flops


# decode
def compute_llm_runtime_gpu_roofline_model(
    sequence_length=128,
    input_prompt_length=0,
    hidden_dim=768,
    num_attn_heads=12,
    gqa_factor=1,
    proj_factor=4,
    num_decoder_layers=12,
    include_classifier=False,
    num_classes=32000,
    batch_size=1
):
    # gqa params
    num_kv_heads = int(num_attn_heads / gqa_factor)
    head_dim = int(hidden_dim / num_attn_heads)

    # q projection
    A = [hidden_dim, hidden_dim]
    B = [hidden_dim, batch_size]
    q_time, _, q_flops, q_mops = compute_mmm_runtime_gpu(A, B)
    q_time = q_time * sequence_length
    q_flops = q_flops * sequence_length
    q_mops = q_mops * sequence_length

    # kv projections (divide by GQA factor)
    A = [hidden_dim / gqa_factor, hidden_dim]
    B = [hidden_dim, batch_size]
    k_time, _, k_flops, k_mops = compute_mmm_runtime_gpu(A, B)
    v_time, _, v_flops, v_mops = compute_mmm_runtime_gpu(A, B)
    k_time = k_time * sequence_length
    v_time = v_time * sequence_length
    k_flops = k_flops * sequence_length
    v_flops = v_flops * sequence_length
    k_mops = k_mops * sequence_length
    v_mops = v_mops * sequence_length

    # rope time
    A = [hidden_dim, batch_size]
    ropeq_time, _, ropeq_flops, ropeq_mops = compute_nonlinear_runtime_gpu(A, operation='rope')
    A = [hidden_dim / gqa_factor, batch_size]
    ropek_time, _, ropek_flops, ropek_mops = compute_nonlinear_runtime_gpu(A, operation='rope')
    ropeq_time = ropeq_time * sequence_length
    ropek_time = ropek_time * sequence_length
    ropeq_flops = ropeq_flops * sequence_length
    ropek_flops = ropek_flops * sequence_length
    ropeq_mops = ropeq_mops * sequence_length
    ropek_mops = ropek_mops * sequence_length

    # compute attention (accounting for input KV cache size)
    qk_time = 0
    smax_time = 0
    score_time = 0
    qk_flops = 0
    smax_flops = 0
    score_flops = 0
    qk_mops = 0
    smax_mops = 0
    score_mops = 0
    for seqlen in range(input_prompt_length + 1, input_prompt_length + sequence_length + 1):
        A = [gqa_factor, head_dim]
        B = [head_dim, seqlen]
        qk_time_tmp, _, qk_flops_tmp, qk_mops_tmp = compute_mmm_runtime_gpu(A, B, nowrite=True)
        qk_time += qk_time_tmp
        qk_flops += qk_flops_tmp
        qk_mops += qk_mops_tmp

        A = [gqa_factor, seqlen]
        smax_time_tmp, _, smax_flops_tmp, smax_mops_tmp = compute_nonlinear_runtime_gpu(A, operation='softmax', nowrite=True, noload=True)
        smax_time += smax_time_tmp
        smax_flops += smax_flops_tmp
        smax_mops += smax_mops_tmp

        A = [gqa_factor, seqlen]
        B = [seqlen, head_dim]
        score_time_tmp, _, score_flops_tmp, score_mops_tmp = compute_mmm_runtime_gpu(A, B, noloadA=True)
        score_time += score_time_tmp
        score_flops += score_flops_tmp
        score_mops += score_mops_tmp

    qk_time = qk_time * num_kv_heads * batch_size
    smax_time = smax_time * num_kv_heads * batch_size
    score_time = score_time * num_kv_heads * batch_size

    qk_flops = qk_flops * num_kv_heads * batch_size
    smax_flops = smax_flops * num_kv_heads * batch_size
    score_flops = score_flops * num_kv_heads * batch_size

    qk_mops = qk_mops * num_kv_heads * batch_size
    smax_mops = smax_mops * num_kv_heads * batch_size
    score_mops = score_mops * num_kv_heads * batch_size

    # compute output projection runtime
    A = [hidden_dim, hidden_dim]
    B = [hidden_dim, batch_size]
    o_time, _, o_flops, o_mops = compute_mmm_runtime_gpu(A, B)
    o_time = o_time * sequence_length
    o_flops = o_flops * sequence_length
    o_mops = o_mops * sequence_length

    # compute layernorm + add runtime
    A = [batch_size, hidden_dim]
    ln1_time, _, ln1_flops, ln1_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd1_time, _, eadd1_flops, eadd1_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')
    ln1_time = ln1_time * sequence_length
    eadd1_time = eadd1_time * sequence_length
    ln1_flops = ln1_flops * sequence_length
    eadd1_flops = eadd1_flops * sequence_length
    ln1_mops = ln1_mops * sequence_length
    eadd1_mops = eadd1_mops * sequence_length

    # compute fc1 runtime
    A = [proj_factor * hidden_dim, hidden_dim]
    B = [hidden_dim, batch_size]
    ffn1_time, _, ffn1_flops, ffn1_mops = compute_mmm_runtime_gpu(A, B)
    ffn1_time = ffn1_time * sequence_length
    ffn1_flops = ffn1_flops * sequence_length
    ffn1_mops = ffn1_mops * sequence_length

    # compute fc1 runtime
    A = [proj_factor * hidden_dim, hidden_dim]
    B = [hidden_dim, batch_size]
    gate_time, _, gate_flops, gate_mops = compute_mmm_runtime_gpu(A, B)
    gate_time = gate_time * sequence_length
    gate_flops = gate_flops * sequence_length
    gate_mops = gate_mops * sequence_length

    # compute sigmoid runtime
    A = [batch_size, proj_factor * hidden_dim]
    sigmoid_time, _, sigmoid_flops, sigmoid_mops = compute_nonlinear_runtime_gpu(A, operation='sigmoid')
    sigmoid_time = sigmoid_time * sequence_length
    sigmoid_flops = sigmoid_flops * sequence_length
    sigmoid_mops = sigmoid_mops * sequence_length

    # compute elemmul runtime
    A = [batch_size, proj_factor * hidden_dim]
    elemmul_time, _, elemmul_flops, elemmul_mops = compute_nonlinear_runtime_gpu(A, operation='elemmul')
    elemmul_time = elemmul_time * sequence_length
    elemmul_flops = elemmul_flops * sequence_length
    elemmul_mops = elemmul_mops * sequence_length

    # compute fc2 runtime
    A = [hidden_dim, proj_factor * hidden_dim]
    B = [proj_factor * hidden_dim, batch_size]
    ffn2_time, _, ffn2_flops, ffn2_mops = compute_mmm_runtime_gpu(A, B)
    ffn2_time = ffn2_time * sequence_length
    ffn2_flops = ffn2_flops * sequence_length
    ffn2_mops = ffn2_mops * sequence_length

    # compute layernorm + add runtime
    A = [batch_size, hidden_dim]
    ln2_time, _, ln2_flops, ln2_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    eadd2_time, _, eadd2_flops, eadd2_mops = compute_nonlinear_runtime_gpu(A, operation='elemadd')
    ln2_time = ln2_time * sequence_length
    eadd2_time = eadd2_time * sequence_length
    ln2_flops = ln2_flops * sequence_length
    eadd2_flops = eadd2_flops * sequence_length
    ln2_mops = ln2_mops * sequence_length
    eadd2_mops = eadd2_mops * sequence_length

    # final layernorm
    A = [batch_size, hidden_dim]
    lnf_time, _, lnf_flops, lnf_mops = compute_nonlinear_runtime_gpu(A, operation='layernorm')
    lnf_time = lnf_time * sequence_length
    lnf_flops = lnf_flops * sequence_length
    lnf_mops = lnf_mops * sequence_length

    # compute classifier runtime
    if include_classifier:
        A = [num_classes, hidden_dim]
        B = [hidden_dim, batch_size]
        emb_time, _, emb_flops, emb_mops = compute_mmm_runtime_gpu(A, B)
        emb_time = emb_time * sequence_length
        emb_flops = emb_flops * sequence_length
        emb_mops = emb_mops * sequence_length
    else:
        emb_time = 0

    # compute total runtime
    ffn_linear_time = (ffn1_time + gate_time + ffn2_time) * num_decoder_layers
    self_mha_linear_time = (q_time + k_time + v_time + o_time) * num_decoder_layers
    self_mha_act2act_time = (qk_time + score_time) * num_decoder_layers
    nonlinear_time = (smax_time + ln1_time + ln2_time + eadd1_time + eadd2_time + sigmoid_time + elemmul_time + ropeq_time + ropek_time) * num_decoder_layers + lnf_time
    total_runtime = ffn_linear_time + self_mha_linear_time + self_mha_act2act_time + nonlinear_time + emb_time

    # compute ai - flops / mops
    ffn_linear_flops = (ffn1_flops + gate_flops + ffn2_flops) * num_decoder_layers
    mha_linear_flops = (q_flops + k_flops + v_flops + o_flops) * num_decoder_layers
    mha_act2act_flops = (qk_flops + score_flops) * num_decoder_layers
    nonlinear_flops = (smax_flops + ln1_flops + ln2_flops + eadd1_flops + eadd2_flops + sigmoid_flops + elemmul_flops + ropeq_flops + ropek_flops) * num_decoder_layers + lnf_flops
    total_flops = ffn_linear_flops + mha_linear_flops + mha_act2act_flops + nonlinear_flops + emb_flops

    ffn_linear_mops = (ffn1_mops + gate_mops + ffn2_mops) * num_decoder_layers
    mha_linear_mops = (q_mops + k_mops + v_mops + o_mops) * num_decoder_layers
    mha_act2act_mops = (qk_mops + score_mops) * num_decoder_layers
    nonlinear_mops = (smax_mops + ln1_mops + ln2_mops + eadd1_mops + eadd2_mops + sigmoid_mops + elemmul_mops + ropeq_mops + ropek_mops) * num_decoder_layers + lnf_mops
    total_mops = ffn_linear_mops + mha_linear_mops + mha_act2act_mops + nonlinear_mops + emb_mops

    # compute AI for linear and attention ops
    linear_ai = (ffn_linear_flops + mha_linear_flops + emb_flops) / (ffn_linear_mops + mha_linear_mops + emb_mops)
    attention_ai = mha_act2act_flops / mha_act2act_mops

    # aggregate AI
    aggregate_ai = total_flops / total_mops

    return total_runtime, ffn_linear_time, self_mha_linear_time, self_mha_act2act_time, nonlinear_time, emb_time, linear_ai, attention_ai, aggregate_ai, total_flops