


class OlmoeParallelConfig():
    def __init__(
        self,
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=None,
        rms_norm_eps=1e-05,
        max_position_embeddings=4096,
        initializer_range=0.02,
        padding_idx=1,

        rope_scaling=None,
        rope_theta=10000.0,
        
        num_experts=64,
        num_experts_per_tok=8,
        norm_topk_prob=False,
        router_aux_loss_coef=0.01,
        output_router_logits=False,
        use_local_router_aux_loss=False,

        use_cache=False,

        pipeline_parallelism=1,
        expert_parallelism=1,
        tensor_parallelism=1,
        virtual_pipeline_parallelism=1,
        use_activation_checkpointing=False,
        activation_checkpointing_level=0,
        use_fast_moe=False,
        use_merged_mlp_in_fast_moe=False,
        use_triton_path_for_gemm_in_fast_moe=False,
        use_triton_path_for_nongemm_in_fast_moe=False,
        # Debug
        clip_qkv=None,
        use_qk_norm=True,
        skip_qk_norm=False,
        force_uniform_routing=False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads

        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.padding_idx = padding_idx

        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_local_router_aux_loss = use_local_router_aux_loss

        self.use_cache = use_cache

        self.pipeline_parallelism = pipeline_parallelism
        self.expert_parallelism = expert_parallelism
        self.tensor_parallelism = tensor_parallelism
        self.virtual_pipeline_parallelism = virtual_pipeline_parallelism

        self.use_activation_checkpointing = use_activation_checkpointing
        self.activation_checkpointing_level = activation_checkpointing_level
        self.use_fast_moe = use_fast_moe
        self.use_merged_mlp_in_fast_moe = use_merged_mlp_in_fast_moe
        self.use_triton_path_for_gemm_in_fast_moe = use_triton_path_for_gemm_in_fast_moe
        self.use_triton_path_for_nongemm_in_fast_moe = use_triton_path_for_nongemm_in_fast_moe

        self.clip_qkv = clip_qkv
        self.use_qk_norm = use_qk_norm
        self.skip_qk_norm = skip_qk_norm
        self.force_uniform_routing = force_uniform_routing

        # Checks
        assert self.num_attention_heads % self.num_key_value_heads == 0
        
    def check_compatibility(self):
        assert (self.num_hidden_layers % self.pipeline_parallelism) == 0
        assert (self.num_hidden_layers // self.pipeline_parallelism) % self.virtual_pipeline_parallelism == 0, f"(num_hidden_layers / pipeline_parallelism) {self.num_hidden_layers // self.pipeline_parallelism} should be divisible by virtual_pipeline_parallelism {self.virtual_pipeline_parallelism}"
        assert (self.num_experts % self.expert_parallelism) == 0, f"num_experts {self.num_experts} should be divisible by expert_parallelism {self.expert_parallelism}"
        assert (self.tensor_parallelism == 1)        

    def __repr__(self):
        config_dict = self.__dict__
        lines = ['OlmoeParallelConfig : {']
        for k, v in config_dict.items():
            lines.append(f'    "{k}": {repr(v)},')
        lines.append('}')
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

__all__ = ["OlmoeParallelConfig"]