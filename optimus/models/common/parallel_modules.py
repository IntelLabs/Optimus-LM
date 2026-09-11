import torch
import torch.nn as nn
import math

from .comms_ptfunctions import forward_allreduce_backward_identity
from .comms_ptfunctions import forward_allgather_backward_pick
from .comms_ptfunctions import forward_inner_split_backward_allgather

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=None, tensor_parallelism=1, tp_ind=0, group=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.group = group

        # Padding embedding table to be able to divide across ranks
        self.vocab_size_per_rank = (vocab_size + tensor_parallelism - 1) // tensor_parallelism
        self.vocab_size_padded = tensor_parallelism * self.vocab_size_per_rank

        self.start_embedding_id = tp_ind * self.vocab_size_per_rank # Note : Start is zero based index
        self.end_embedding_id = ((tp_ind + 1) * self.vocab_size_per_rank) # Note: End is one based index

        self.weight = nn.Parameter(torch.empty((self.vocab_size_per_rank, embedding_size)))

        self.local_padding_idx = None
        if (padding_idx is not None) and (padding_idx >= self.start_embedding_id) and (padding_idx < self.end_embedding_id):
            self.local_padding_idx = padding_idx - self.start_embedding_id
    
    def forward(self, input):
        if self.tensor_parallelism > 1:
            input_mask = (input < self.start_embedding_id) | (input >= self.end_embedding_id)
            masked_input = input.clone() - self.start_embedding_id
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output = torch.nn.functional.embedding(masked_input, self.weight, self.local_padding_idx)

        if self.tensor_parallelism > 1:
            output[input_mask, :] = 0
            output = forward_allreduce_backward_identity.apply(output, self.group)
        return output
    
    def set_parameters_from_full_module(self, module):
        start_id = self.start_embedding_id
        end_id = min(self.end_embedding_id, self.vocab_size)
        start_id_p = start_id - self.start_embedding_id
        end_id_p = end_id - self.start_embedding_id
        with torch.no_grad():
            self.weight[start_id_p:end_id_p].copy_(module.weight[start_id:end_id])

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=1)

class ParallelLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False, tensor_parallelism=1, tp_ind=0, group=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.group = group

        assert bias == False, "ParallelLMHead does not support bias currently."

        self.hidden_size_per_rank = self.hidden_size // tensor_parallelism
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.hidden_size_per_rank)))
        self.bias = None

    def forward(self, hidden_states):
        if self.tensor_parallelism > 1:
            hidden_states = forward_inner_split_backward_allgather.apply(hidden_states, self.tp_ind, self.tensor_parallelism, self.group)
        
        logits = nn.functional.linear(hidden_states, self.weight, self.bias)

        if self.tensor_parallelism > 1:
            logits = forward_allreduce_backward_identity.apply(logits, self.group)
        
        return logits
    
    def set_parameters_from_full_module(self, module):
        with torch.no_grad():
            self.weight.copy_(module.weight[:, self.tp_ind * self.hidden_size_per_rank:(self.tp_ind + 1) * self.hidden_size_per_rank])

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=1)