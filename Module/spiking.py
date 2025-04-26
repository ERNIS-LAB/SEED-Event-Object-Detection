import torch

class SpikeFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
        
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dE_dz = grad_output

        dz_du = dampening_factor * torch.maximum(1 - pseudo_derivative_support * torch.abs(
            inp), torch.Tensor((0,)).to(grad_output.device))

        dE_dv = dE_dz * dz_du
        return dE_dv, None, None
    
    
class SpikeFunctionRectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
       
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dE_dz = grad_output

        dz_du = torch.where((inp < 3.) and (inp > -3.), torch.ones_like(inp), torch.zeros_like(inp))

        dE_dv = dE_dz * dz_du
        return dE_dv, None, None