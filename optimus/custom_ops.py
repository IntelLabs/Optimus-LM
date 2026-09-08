import torch
import pcl_xpu_customops

class IndexAddOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, final_output, indices):
        pcl_xpu_customops.index_add_op(expert_output, final_output, indices)
        ctx.save_for_backward(indices, expert_output)
        return final_output

    @staticmethod
    def backward(ctx, grad_output):
        indices, expert_output = ctx.saved_tensors
        expert_output_grad = torch.zeros_like(expert_output)
        pcl_xpu_customops.index_gather_op(grad_output, expert_output_grad, indices)
        return expert_output_grad, None, None