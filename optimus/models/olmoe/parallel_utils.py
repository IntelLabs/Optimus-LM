import torch

def calculate_grad_norm_for_tensor_parallelism(model, tensor_parallelism=1, tp_group=None, 
    tp_degree=15, tensor_parallelism_moe_choice=1, disable_combined_qknorm=False):
    assert tp_degree < 16, "TP degree should be less than 16"
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    params_sharded = []

    if (tp_degree & 1) == 1:
        params_sharded.append("model.embed_tokens.weight")

    if (tp_degree & 2) == 2:
        for i in range(len(model.model.layers)):
            params_sharded.append(f"model.layers.{i}.self_attn.q_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.k_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.v_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.o_proj.weight")
            if not disable_combined_qknorm:
                params_sharded.append(f"model.layers.{i}.self_attn.q_norm.weight")
                params_sharded.append(f"model.layers.{i}.self_attn.k_norm.weight")

    if (tp_degree & 4) == 4:
        for i in range(len(model.model.layers)):
            for e in range(model.model.layers[i].mlp.num_experts):
                params_sharded.append(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight")
                params_sharded.append(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight")
                params_sharded.append(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight")

    if (tp_degree & 8) == 8:
        params_sharded.append("lm_head.weight")

    # Grouping gradients
    params_replicated = []
    grads_sharded = []
    grads_replicated = []
    for name,p in model.named_parameters():
        if p.grad is not None:
            if name in params_sharded:
                grads_sharded.append(p.grad)
            else:
                grads_replicated.append(p.grad)
                params_replicated.append(name)
    
    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     for _ in params_replicated:
    #         print(_,flush=True)

    # Calculate norm
    grad_norm_sharded = torch.nn.utils.get_total_norm(grads_sharded).to(torch.float32)
    grad_norm_replicated = torch.nn.utils.get_total_norm(grads_replicated).to(torch.float32)

    # Undo sqrt
    grad_norm_sharded_sq = torch.pow(grad_norm_sharded, 2)
    grad_norm_replicated_sq = torch.pow(grad_norm_replicated, 2)

    # Default grad_norm
    if torch.distributed.is_initialized() and tensor_parallelism > 1:        
        grad_norm_sq = grad_norm_sharded_sq + grad_norm_replicated_sq / tensor_parallelism
        torch.distributed.all_reduce(grad_norm_sq, group=tp_group)
        grad_norm = grad_norm_sq.sqrt()
    else:
        grad_norm = torch.sqrt(grad_norm_sharded_sq + grad_norm_replicated_sq)

    return grad_norm
