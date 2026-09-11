

class OlmoParallelConfig():
    def __init__(
        self,
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        clip_qkv=None,
        # Additional arguments for parallel OLMo
        use_rms_norm=False,
        use_qk_norm=False,
        skip_qk_norm=True,
        tensor_parallelism=1,
        pipeline_parallelism=1,
        virtual_pipeline_parallelism=1,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clip_qkv = clip_qkv

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Parallelism configuration
        self.use_rms_norm = use_rms_norm
        self.use_qk_norm = use_qk_norm
        self.skip_qk_norm = skip_qk_norm
        self.tensor_parallelism = tensor_parallelism
        self.pipeline_parallelism = pipeline_parallelism
        self.virtual_pipeline_parallelism = virtual_pipeline_parallelism

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
    
    def check_compatibility(self):
        assert (self.intermediate_size % self.tensor_parallelism) == 0, f"intermediate_size {self.intermediate_size} should be divisible by tensor_parallelism {self.tensor_parallelism}"
        assert (self.num_attention_heads % self.tensor_parallelism) == 0, f"num_attention_heads {self.num_attention_heads} should be divisible by tensor_parallelism {self.tensor_parallelism}"
        assert (self.num_hidden_layers % self.pipeline_parallelism) == 0, f"num_hidden_layers {self.num_hidden_layers} should be divisible by pipeline_parallelism {self.pipeline_parallelism}"
        assert (self.num_hidden_layers // self.pipeline_parallelism) % self.virtual_pipeline_parallelism == 0, f"(num_hidden_layers / pipeline_parallelism) {self.num_hidden_layers // self.pipeline_parallelism} should be divisible by virtual_pipeline_parallelism {self.virtual_pipeline_parallelism}"

    def __repr__(self):
        config_dict = self.__dict__
        lines = ['OlmoParallelConfig : {']
        for k, v in config_dict.items():
            lines.append(f'    "{k}": {repr(v)},')
        lines.append('}')
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()   