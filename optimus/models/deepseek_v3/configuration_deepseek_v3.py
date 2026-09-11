class DeepseekV3ParallelConfig():
    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=128,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        n_group=8,
        topk_group=4,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_interleave=True,
        pad_token_id=None,
        # Parallelism
        expert_parallelism=1,
        pipeline_parallelism=1,
        tensor_parallelism=1,
        virtual_pipeline_parallelism=1,
        use_activation_checkpointing=False,
        activation_checkpointing_level=0, # Bit 0 for MoE, Bit 1 for Attention, Bit 2 for Norm
        # Debug
        force_uniform_routing=False,
        ):
    
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_interleave = rope_interleave
        self.pad_token_id = pad_token_id

        self.head_dim = self.qk_rope_head_dim
        self.num_key_value_heads = self.num_attention_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Hard coding to DSV3 config values to avoid complexity of handling multiple cases in the code
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        }
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.rope_theta = 10000

        self.use_cache = False

        # Parallelism related
        self.expert_parallelism = expert_parallelism
        self.pipeline_parallelism = pipeline_parallelism
        self.tensor_parallelism = tensor_parallelism
        self.virtual_pipeline_parallelism = virtual_pipeline_parallelism

        self.use_activation_checkpointing = use_activation_checkpointing
        self.activation_checkpointing_level = activation_checkpointing_level

        # Debug related
        self.force_uniform_routing = force_uniform_routing
    
    def __repr__(self):
        config_dict = self.__dict__
        lines = ['DeepseekV3ParallelConfig : {']
        for k, v in config_dict.items():
            lines.append(f'    "{k}": {repr(v)},')
        lines.append('}')
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

