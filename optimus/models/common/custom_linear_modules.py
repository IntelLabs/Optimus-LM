import math

import torch
import torch.nn as nn

USE_EXPLICIT_MM_CALLS = False

class PclLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PclLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
            # self.bias = nn.Parameter('bias', None)

    def forward(self, input):
        if USE_EXPLICIT_MM_CALLS:
            output = torch.matmul(input, self.weight.t())
            if self.bias != None:
                output = output + self.bias
            return output
        else:
            output = nn.functional.linear(input, self.weight, self.bias)
            return output
            """
            size_out = input.size()[:-1] + (self.out_features,)
            if self.bias != None:
                output = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight.t())
            else:
                output = torch.matmul(input.view(-1, input.size(-1)), self.weight.t())
            output = output.view(size_out)
            return output
            """

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class ColumnLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tensor_parallelism=1):
        super(ColumnLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallelism = tensor_parallelism
        self.out_features_per_rank = self.out_features//tensor_parallelism
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            # self.bias = nn.Parameter('bias', None)
            self.bias = None
        
    def forward(self, input):
        if USE_EXPLICIT_MM_CALLS:
            output = torch.matmul(input, self.weight.t())
            if self.bias != None:
                output = output + self.bias
            return output
        else:
            output = nn.functional.linear(input, self.weight, self.bias)
            return output
            """
            size_out = input.size()[:-1] + (self.out_features_per_rank,)
            if self.bias != None:
                output = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight.t())
            else:
                output = torch.matmul(input.view(-1, input.size(-1)), self.weight.t())
            output = output.view(size_out)
            return output
            """
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class RowLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tensor_parallelism=1, skip_bias_add=False):
        super(RowLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallelism = tensor_parallelism
        self.in_features_per_rank = self.in_features//tensor_parallelism
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # self.bias = nn.Parameter('bias', None)
            self.bias = None
        self.skip_bias_add = skip_bias_add
        
    def forward(self, input):
        if USE_EXPLICIT_MM_CALLS:
            output = torch.matmul(input, self.weight.t())
            if self.bias != None and (not self.skip_bias_add):
                output = output + self.bias
            return output
        else:
            output = nn.functional.linear(input, self.weight, self.bias)
            return output
            """
            size_out = input.size()[:-1] + (self.out_features,)
            if not self.skip_bias_add:
                if self.bias != None:
                    output = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight.t())
                else:
                    output = torch.matmul(input.view(-1, input.size(-1)), self.weight.t())
            else:
                output = torch.matmul(input.view(-1, input.size(-1)), self.weight.t())
            output = output.view(size_out)
            return output
            """

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

