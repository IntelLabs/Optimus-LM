import torch

from typing import Optional, Tuple, Union

# No baggage version of HuggingFace's load balancing loss function
def single_input_load_balancing_loss_func(gate_logits, num_experts, top_k):
    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

# Layer-level load balancing loss function
def layer_level_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, int]:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    assert attention_mask is None

    stacked_gate_logits = torch.stack([layer_gate for layer_gate in gate_logits], dim=0)
    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=1)
    router_prob_per_expert = torch.mean(routing_weights, dim=1)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(1))
    return overall_loss * num_experts


