# Adapted from transformers.modeling_rope_utils (v4.57.6) to avoid a hard dependency on transformers.
import logging
from functools import wraps
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).
    """

    def dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper


def _compute_default_rope_parameters(config, device=None, seq_len=None):
    """Computes the inverse frequencies according to the original RoPE implementation."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(config, device=None, seq_len=None):
    """Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev."""
    factor = config.rope_scaling["factor"]

    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(config, device=None, seq_len=None):
    """Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


# Maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters.
# Only "default", "linear", and "dynamic" are supported by the configs in this repo.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


# Like `ROPE_INIT_FUNCTIONS`, only "default", "linear", and "dynamic" are supported by the configs in this repo.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
}


def rope_config_validation(config, ignore_keys: Optional[set] = None):
    """Validate the RoPE config arguments, given a config object"""
    rope_scaling = getattr(config, "rope_scaling", None)  # not a default parameter in `PretrainedConfig`
    if rope_scaling is None:
        return

    # BC: "rope_type" was originally "type"
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config, ignore_keys=ignore_keys)
    else:
        logger.warning(
            f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
        )
