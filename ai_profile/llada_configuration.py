import torch
from transformers import AutoModel, AutoTokenizer

def analyze_llada_model():
    """Load and analyze LLaDA model structure."""
    
    LLADA_MODEL_ID = 'GSAI-ML/LLaDA-8B-Instruct'
    
    print("=" * 60)
    print(f"LLaDA Model Structure Analysis: {LLADA_MODEL_ID}")
    print("=" * 60)
    
    try:
        # Load model and tokenizer
        print("\n1. Loading model...")
        model = AutoModel.from_pretrained(
            LLADA_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(LLADA_MODEL_ID, trust_remote_code=True)
        print("✅ Model loaded successfully!")
        
        model = model.cpu()
        
        # Basic model info
        print(f"\n2. Model Info:")
        print(f"   - Type: {type(model).__name__}")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Model structure
        print(f"\n3. Model Structure:")
        analyze_structure(model)
        
        # Tokenizer info
        print(f"\n4. Tokenizer Info:")
        print(f"   - Vocab size: {tokenizer.vocab_size:,}")
        print(f"   - Special tokens: {tokenizer.special_tokens_map}")
        
        # Config analysis
        print(f"\n5. Model Config:")
        analyze_config(model)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def analyze_structure(model):
    """Analyze model structure without duplication."""
    
    print("📦 Model Architecture:")
    
    # Find the actual transformer blocks
    transformer_blocks = None
    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        if hasattr(model.model.transformer, 'blocks'):
            transformer_blocks = model.model.transformer.blocks
        elif hasattr(model.model.transformer, 'layers'):
            transformer_blocks = model.model.transformer.layers
    
    if transformer_blocks:
        print(f"   └─ Transformer Blocks: {len(transformer_blocks)} layers")
        
        # Analyze first layer in detail
        if len(transformer_blocks) > 0:
            print(f"   └─ First Layer Structure:")
            analyze_transformer_layer(transformer_blocks[0], "   ")
    
    # Show other components
    print(f"   └─ Other Components:")
    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer = model.model.transformer
        
        # Token embedding
        if hasattr(transformer, 'wte'):
            param_count = sum(p.numel() for p in transformer.wte.parameters())
            print(f"      └─ Token Embedding: {param_count:,} params")
        
        # Final layer norm
        if hasattr(transformer, 'ln_f'):
            param_count = sum(p.numel() for p in transformer.ln_f.parameters())
            print(f"      └─ Final LayerNorm: {param_count:,} params")
        
        # Output projection
        if hasattr(transformer, 'ff_out'):
            param_count = sum(p.numel() for p in transformer.ff_out.parameters())
            print(f"      └─ Output Projection: {param_count:,} params")

def analyze_transformer_layer(layer, indent):
    """Analyze transformer layer components."""
    
    print(f"{indent}🔍 Layer Components:")
    
    # Group components by type
    attention_components = []
    ffn_components = []
    norm_components = []
    other_components = []
    
    for name, module in layer.named_modules():
        if name == "":  # Skip the layer itself
            continue
        module_type = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters())
        
        component_info = f"{name}: {module_type} ({param_count:,} params)"
        
        if 'attn' in name.lower() or 'q_proj' in name.lower() or 'k_proj' in name.lower() or 'v_proj' in name.lower():
            attention_components.append(component_info)
        elif 'ff' in name.lower() or 'up_proj' in name.lower():
            ffn_components.append(component_info)
        elif 'norm' in name.lower():
            norm_components.append(component_info)
        else:
            other_components.append(component_info)
    
    # Print attention components
    if attention_components:
        print(f"{indent}   📌 Attention:")
        for comp in attention_components:
            print(f"{indent}      └─ {comp}")
    
    # Print FFN components
    if ffn_components:
        print(f"{indent}   🧠 Feed-Forward Network:")
        for comp in ffn_components:
            print(f"{indent}      └─ {comp}")
    
    # Print normalization components
    if norm_components:
        print(f"{indent}   📏 Layer Normalization:")
        for comp in norm_components:
            print(f"{indent}      └─ {comp}")
    
    # Print other components
    if other_components:
        print(f"{indent}   🔧 Other:")
        for comp in other_components:
            print(f"{indent}      └─ {comp}")

def analyze_config(model):
    """Analyze model configuration."""
    
    if hasattr(model, 'config'):
        config = model.config
        print(f"   - Model type: {getattr(config, 'model_type', 'N/A')}")
        print(f"   - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"   - Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")
        print(f"   - Attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"   - KV heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
        print(f"   - Layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"   - Max position: {getattr(config, 'max_position_embeddings', 'N/A')}")
        print(f"   - Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"   - Activation: {getattr(config, 'hidden_act', 'N/A')}")
        
        # LLaDA specific
        if hasattr(config, 'use_parallel_residual'):
            print(f"   - Parallel residual: {config.use_parallel_residual}")
        if hasattr(config, 'rope_theta'):
            print(f"   - RoPE theta: {config.rope_theta}")

def main():
    """Main execution function."""
    
    print("🚀 Starting LLaDA model analysis...")
    
    model, tokenizer = analyze_llada_model()
    
    if model is not None:
        print(f"\n" + "=" * 60)
        print("✅ LLaDA model analysis completed!")
        print("=" * 60)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("❌ Model analysis failed")

if __name__ == "__main__":
    main()
