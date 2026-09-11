import re

def get_modified_config(model_choice, config):
    hf_model_choice, override_config_str = model_choice, None
    if len(model_choice.split("--")) == 2:
        hf_model_choice, override_config_str = model_choice.split("--")
     
    # Common for all models
    config.use_cache = False
    
    if hf_model_choice in ["deepseek-ai/DeepSeek-V3"]:
        # L[0-9]+  -> Sets num_hidden_layers to L
        # DL[0-9]+ -> Sets first_k_dense_replace to DL
        # H[0-9]+  -> Sets hidden_size to H
        # I[0-9]+  -> Sets intermediate_size to I
        # MI[0-9]+ -> Sets MoE intermediate size to MI
        # A[0-9]+  -> Sets num_attention_heads to A
        # N[0-9]+  -> Sets n_routed_experts to N
        # K[0-9]+  -> Sets num_experts_per_tok to K
        # SE[0-9]+ -> Sets n_shared_experts to SE

        tokens = override_config_str.split("-") if override_config_str is not None else []

        change_dict = {
            'L': config.num_hidden_layers,
            'DL': config.first_k_dense_replace,
            'H': config.hidden_size,
            'I': config.intermediate_size,
            'MI': config.moe_intermediate_size,
            'A': config.num_attention_heads,
            'N': config.n_routed_experts,
            'K': config.num_experts_per_tok,
            'SE': config.n_shared_experts
            }
        for token in tokens:
            if token.startswith("L"):
                change_dict['L'] = int(token.replace("L", ""))
            elif token.startswith("DL"):
                change_dict['DL'] = int(token.replace("DL", ""))
            elif token.startswith("H"):
                change_dict['H'] = int(token.replace("H", ""))
            elif token.startswith("I"):
                change_dict['I'] = int(token.replace("I", ""))
            elif token.startswith("MI"):
                change_dict['MI'] = int(token.replace("MI", ""))
            elif token.startswith("N"):
                change_dict['N'] = int(token.replace("N", ""))
            elif token.startswith("K"):
                change_dict['K'] = int(token.replace("K", ""))
            elif token.startswith("SE"):
                change_dict['SE'] = int(token.replace("SE", ""))
            elif token.startswith("A"):
                change_dict['A'] = int(token.replace("A", ""))
            
        config.num_hidden_layers = change_dict['L']
        config.first_k_dense_replace = change_dict['DL']
        config.hidden_size = change_dict['H']
        config.intermediate_size = change_dict['I']
        config.moe_intermediate_size = change_dict['MI']
        config.n_routed_experts = change_dict['N']
        config.num_experts_per_tok = change_dict['K']
        config.n_shared_experts = change_dict['SE']
        config.num_attention_heads = change_dict['A']
        config.num_key_value_heads = change_dict['A'] 

    elif hf_model_choice in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B"]:
        config.max_position_embeddings = 4096
        config.tie_word_embeddings = False
        # L[0-9] -> Sets num_hidden_layers to L
        # S[0-9] -> Scales (num_attention_heads, num_key_value_heads, hidden_size, intermediate_size) by S

        tokens = override_config_str.split("-") if override_config_str is not None else []

        change_dict = {
            'L': config.num_hidden_layers,
            'H': config.hidden_size,
            'I': config.intermediate_size
            }
        
        for token in tokens:
            if token.startswith("L"):
                change_dict['L'] = int(token.replace("L", ""))
            elif token.startswith("H"):
                change_dict['H'] = int(token.replace("H", ""))
            elif token.startswith("I"):
                change_dict['I'] = int(token.replace("I", ""))
        
        # Effect of L
        config.num_hidden_layers = change_dict['L']
        # Effect of H
        config.hidden_size = change_dict['H']
        # Effect of I
        config.intermediate_size = change_dict['I']

        # Adjusting head_dim as H is changing
        config.head_dim = config.hidden_size // config.num_attention_heads

        # # Effect of S
        # scale = change_dict['S']
        # config.num_attention_heads = int(config.num_attention_heads * scale)
        # config.num_key_value_heads = int(config.num_key_value_heads * scale)
        # config.hidden_size = int(config.hidden_size * scale)
        # config.intermediate_size = int(config.intermediate_size * scale)

    elif hf_model_choice in ["allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf"]:
        config.max_position_embeddings = 4096
        config.tie_word_embeddings = False
        # L[0-9] -> Sets num_hidden_layers to L
        # S[0-9] -> Scales (num_attention_heads, num_key_value_heads, hidden_size, intermediate_size) by S

        tokens = override_config_str.split("-") if override_config_str is not None else []

        change_dict = {
            'L': config.num_hidden_layers,
            'S': 1.0
            }
        
        for token in tokens:
            if token.startswith("L"):
                change_dict['L'] = int(token.replace("L", ""))
            elif token.startswith("S"):
                change_dict['S'] = float(token.replace("S", ""))
            elif token.startswith('TWE'):
                config.tie_word_embeddings = True
            elif token.startswith('RMS'):
                config.use_rms_norm = True
        
        # Effect of L
        config.num_hidden_layers = change_dict['L']

        # Effect of S
        scale = change_dict['S']
        config.num_attention_heads = int(config.num_attention_heads * scale)
        config.num_key_value_heads = int(config.num_key_value_heads * scale)
        config.hidden_size = int(config.hidden_size * scale)
        config.intermediate_size = int(config.intermediate_size * scale)

    elif hf_model_choice == "allenai/OLMoE-1B-7B-0924":
        
        tokens = override_config_str.split("-") if override_config_str is not None else []
        apply_dict = {
            'L'  : config.num_hidden_layers, # Overrides number of layers
            'SA' : 1.0, # Scales attention heads
            'SH' : 1.0, # Scales hidden size
            'SE' : 1.0, # Scales experts
            'SC' : 1.0, # Scales num_experts_per_token
            'SI' : 1.0, # Scales intermediate size
            'NAL': False, # No Auxiliary Loss
            'SKIPQKNORM' : False, # Skip qk norm
            }

        # 'SM'  : 1.0, # SA U SH
        # 'S'   : 1.0, # SA U SH U SE U SI
        valid_keys = list(apply_dict.keys()) + ["SM", "S"]

        # Breaking down the modify string
        request_dict = {}
        for token in tokens:
            key, value = re.match(r"([A-Z]+)([0-9.]*)", token).groups()
            assert key in valid_keys, f"{key} should be among {valid_keys}"
            if len(value) != 0:
                request_dict[key] = float(value)
            else:
                request_dict[key] = True
        
        if "S" in request_dict:
            assert not ("SA" in request_dict or "SH" in request_dict or "SE" in request_dict or "SI" in request_dict), "If S is provided, then SA,SH, SI and SE should not be provided"

        if "SM" in request_dict:
            assert not ("SA" in request_dict or "SH" in request_dict), "If SM is provided, then SA and SH should not be provided"

        # Modify apply_dict based on request_dict
        for key,value in request_dict.items():
            if key == "S":
                apply_dict["SA"] = value
                apply_dict["SH"] = value
                apply_dict["SE"] = value
                apply_dict["SI"] = value
            elif key == "SM":
                apply_dict["SA"] = value
                apply_dict["SH"] = value
            else:
                apply_dict[key] = value

        # Apply the changes
        config.num_hidden_layers    = int(apply_dict['L'])
        config.num_attention_heads  = int(apply_dict['SA'] * config.num_attention_heads)
        config.num_key_value_heads  = int(apply_dict['SA'] * config.num_key_value_heads)
        config.hidden_size          = int(apply_dict['SH'] * config.hidden_size)
        config.num_experts          = int(apply_dict['SE'] * config.num_experts)
        config.num_experts_per_tok  = int(apply_dict['SC'] * config.num_experts_per_tok)
        config.intermediate_size    = int(apply_dict['SI'] * config.intermediate_size)
        config.output_router_logits = not apply_dict['NAL']
        config.skip_qk_norm          = apply_dict['SKIPQKNORM']

    return config

if __name__ == "__main__":
    hf_model_choice = "allenai/OLMoE-1B-7B-0924"
    # modify_tag = "SM1.5-SE1.125" # Crawl
    # modify_tag = "SM2.25-SE1.875-L64" # Walk
    # modify_tag = "SM3-SE6-SI1.5-L96" # Run
    # modify_tag = "SM1.5-SE1.125" # Crawl
    modify_tag = "SA3-SH1.5-SE2.25-SC0.5-SI2-L36"
    model_choice = f"{hf_model_choice}--{modify_tag}"

    from transformers import AutoConfig
    # Original
    config_hf = AutoConfig.from_pretrained(hf_model_choice)
    # Modified
    config = AutoConfig.from_pretrained(hf_model_choice)
    config = get_modified_config(model_choice, config)

    print(f"Tag : {modify_tag}")
    print(f"Key :  OLD  NEW")
    print(f"L   : {config_hf.num_hidden_layers:4d} {config.num_hidden_layers:4d}")
    print(f"AQ  : {config_hf.num_attention_heads:4d} {config.num_attention_heads:4d}")
    print(f"AKV : {config_hf.num_key_value_heads:4d} {config.num_key_value_heads:4d}")
    print(f"H   : {config_hf.hidden_size:4d} {config.hidden_size:4d}")
    print(f"E   : {config_hf.num_experts:4d} {config.num_experts:4d}")
    print(f"C   : {config_hf.num_experts_per_tok:4d} {config.num_experts_per_tok:4d}")
    print(f"I   : {config_hf.intermediate_size:4d} {config.intermediate_size:4d}")
    print(f"RAL : {config_hf.output_router_logits} {config.output_router_logits}")

    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    with init_empty_weights():
        modified_config_hf = get_modified_config(model_choice, config_hf)
        model = AutoModelForCausalLM.from_config(config)

    # for name,param in model.named_parameters():
    #     print(name, param.shape, param.numel())
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters : {param_size * 1e-9:.2f} B ({param_size})")
