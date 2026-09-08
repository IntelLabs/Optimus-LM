import torch

def populate_ColumnLinear_from_Linear(linear_module, col_linear_module, rank, is_weight_oxi=True, tensor_parallelism=None):
    """
    shape(Linear.weight) = (out_features, in_features)
    shape(Linear.bias) = (out_features)

    shape(ColumnLinear.weight) = (out_features/tensor_parallelism, in_features)
    shape(ColumnLinear.bias) = (out_features/tensor_parallelism)
    """
    out_features_per_rank = col_linear_module.out_features_per_rank if tensor_parallelism == None else (linear_module.out_features // tensor_parallelism)
    with torch.no_grad():
        # Weight copy
        if is_weight_oxi:
            weight_p = linear_module.weight.data[rank*out_features_per_rank:(rank+1)*out_features_per_rank]
        else:
            weight_p = linear_module.weight.t().data[rank*out_features_per_rank:(rank+1)*out_features_per_rank]
        col_linear_module.weight.data.copy_(weight_p)

        # Bias copy
        assert ((linear_module.bias == None) ^ (col_linear_module.bias == None)) == False
        if linear_module.bias != None:
            bias_p = linear_module.bias.data[rank*out_features_per_rank:(rank+1)*out_features_per_rank]
            col_linear_module.bias.data.copy_(bias_p)

def populate_RowLinear_from_Linear(linear_module, row_linear_module, rank, is_weight_oxi=True, add_bias_after_allreduce=False, tensor_parallelism=None):
    """
    shape(Linear.weight) = (out_features, in_features)
    shape(Linear.bias) = (out_features)

    shape(RowLinear.weight) = (out_features, in_features / tensor_parallelism)
    shape(RowLinear.bias) = (out_features)
    """
    in_features_per_rank = row_linear_module.in_features_per_rank if tensor_parallelism == None else (linear_module.in_features // tensor_parallelism)
    tensor_parallelism = row_linear_module.tensor_parallelism if tensor_parallelism == None else tensor_parallelism
    with torch.no_grad():
        # Weight copy
        if is_weight_oxi:
            weight_p = linear_module.weight.data[:, rank*in_features_per_rank:(rank+1)*in_features_per_rank]
        else:
            weight_p = linear_module.weight.t().data[:, rank*in_features_per_rank:(rank+1)*in_features_per_rank]
        row_linear_module.weight.data.copy_(weight_p)

        # Bias copy
        assert ((linear_module.bias == None) ^ (row_linear_module.bias == None)) == False
        if linear_module.bias != None:
            bias_p = linear_module.bias if add_bias_after_allreduce else (linear_module.bias / tensor_parallelism)
            row_linear_module.bias.data.copy_(bias_p)

def populate_SplitAttentionLinear_from_AttentionLinear(attn_linear, split_attn_linear, num_heads, head_size, rank, tensor_parallelism, is_weight_oxi=True, layout_mode=0):
    # layout_mode = 0 # Combined linear layer for QKV (num_heads == num_key_value_heads)
    # layout_mode = 1 # Combined linear layer for QKV (num_heads != num_key_value_heads)
    # layout_mode = 2 # Seperate linear layer for Q,K,V
    
    ## layout_mode = 0 # Combined linear layer for QKV (num_heads == num_key_value_heads)
    # shape(AttentionLinear.weight)      = [(3, num_heads,          head_size), hidden_size]
    # shape(SplitAttentionLinear.weight) = [(3, num_heads_per_rank, head_size), hidden_size]

    # layout_mode = 1 # Combined linear layer for QKV (num_heads != num_key_value_heads)
    # shape(AttentionLinear.weight)      = [(num_heads+2*num_key_value_heads,                   head_size), hidden_size]
    # shape(SplitAttentionLinear.weight) = [(num_heads_per_rank+2*num_key_value_heads_per_rank, head_size), hidden_size]

    # layout_mode = 2 # Seperate linear layer for Q,K,V
    # shape(AttentionLinear.weight)      = [(num_heads, head_size), hidden_size]
    # shape(SplitAttentionLinear.weight) = [(num_heads_per_rank, head_size), hidden_size]
    
    
    if layout_mode == 0:
        num_heads_per_rank = num_heads // tensor_parallelism
        with torch.no_grad():
            weight = attn_linear.weight if is_weight_oxi else attn_linear.weight.t()
            hidden_size = attn_linear.weight.shape[1] if is_weight_oxi else attn_linear.weight.shape[0]
            weight = weight.reshape(3, tensor_parallelism, num_heads_per_rank, head_size, hidden_size)
            weight_p = weight[:,rank,:,:,:].reshape((3*num_heads_per_rank*head_size), hidden_size)

            bias = attn_linear.bias
            bias = bias.reshape(3, tensor_parallelism, num_heads_per_rank, head_size)
            bias_p = bias[:,rank,:,:].reshape(3*num_heads_per_rank*head_size)

            split_attn_linear.weight.data.copy_(weight_p)
            split_attn_linear.bias.data.copy_(bias_p)
    elif layout_mode == 1:
        # TODO
        pass
    elif layout_mode == 2:
        num_heads_per_rank = num_heads // tensor_parallelism
        with torch.no_grad():
            # Weight copy
            weight = attn_linear.weight if is_weight_oxi else attn_linear.weight.t()
            hidden_size = attn_linear.weight.shape[1] if is_weight_oxi else attn_linear.weight.shape[0]
            weight = weight.reshape(tensor_parallelism, num_heads_per_rank, head_size, hidden_size)
            weight_p = weight[rank,:,:,:].reshape((num_heads_per_rank*head_size), hidden_size)
            split_attn_linear.weight.data.copy_(weight_p)
            
            # Bias copy
            assert ((attn_linear.bias == None) ^ (split_attn_linear.bias == None)) == False
            if attn_linear.bias != None:
                bias = attn_linear.bias
                bias = bias.reshape(tensor_parallelism, num_heads_per_rank, head_size)
                bias_p = bias[rank,:,:].reshape(num_heads_per_rank*head_size)
                split_attn_linear.bias.data.copy_(bias_p)


