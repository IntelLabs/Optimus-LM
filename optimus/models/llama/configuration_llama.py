from ..common.modeling_rope_utils import rope_config_validation

class LlamaParallelConfig():
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        rope_theta=500000.0,
        rope_scaling=None,
        head_dim=None,
        # Additional arguments for parallel model
        tensor_parallelism=1,
        pipeline_parallelism=1,
        virtual_pipeline_parallelism=1,
        use_activation_checkpointing=False,
        activation_checkpointing_level=0
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
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.tensor_parallelism = tensor_parallelism
        self.pipeline_parallelism = pipeline_parallelism
        self.virtual_pipeline_parallelism = virtual_pipeline_parallelism

        self.use_activation_checkpointing = use_activation_checkpointing
        self.activation_checkpointing_level = activation_checkpointing_level
        assert self.activation_checkpointing_level in [0, 4], "Only activation_checkpointing_level 0 and 4 are supported."
    
    def __repr__(self):
        config_dict = self.__dict__
        lines = ['LlamaParallelConfig : {']
        for k, v in config_dict.items():
            lines.append(f'    "{k}": {repr(v)},')
        lines.append('}')
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()   
