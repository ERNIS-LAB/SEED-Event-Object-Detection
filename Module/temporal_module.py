
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from functools import partial

import math
import torch.nn.functional as F
from .basic_module import ConvLayer, ConditionalConvLayer
from .spiking import SpikeFunction

def time_to_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
   
    t, n, c, h, w = x.size()
    x = x.view(n * t, c, h, w)
    return (x, n)


def batch_to_time(x: torch.Tensor, n: int) -> torch.Tensor:
   
    nt, c, h, w = x.size()
    time = nt // n
    x = x.view(time, n, c, h, w)
    return x


class SequenceWise(nn.Module):
    def __init__(self, module, ndims=5):
      
        super(SequenceWise, self).__init__()
        self.ndims = ndims
        self.module = module

        if hasattr(self.module, "out_channels"):
            self.out_channels = self.module.out_channels

    def forward(self, x):
        if x.dim() == self.ndims - 1:
            return self.module(x)
        else:
            x4, batch_size = time_to_batch(x)
            y4 = self.module(x4)
            return batch_to_time(y4, batch_size)

    def __repr__(self):
        module_str = self.module.__repr__()
        str = f"{self.__class__.__name__} (\n{module_str})"
        return str



class ConvEventLSTMCell(nn.Module):
    
    def __init__(self, hidden_dim, kernel_size, in_size, conv_func=nn.Conv2d):
        super(ConvEventLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=True)

        self.prev_h = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        self.prev_c = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        print(self.thr_reparam.shape)

    @torch.jit.export
    def get_dims_NCHW(self):
        return self.prev_h.size()

    def forward(self, x,stage):
        assert x.dim() == 5

        xseq = x.unbind(0)

        assert self.prev_h.size() == self.prev_c.size()

        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xseq[0].size()
        
        assert input_C == 4 * hidden_C
        assert hidden_C == self.hidden_dim

        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = x.device
            self.prev_h = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)
            self.prev_c = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)

        self.prev_h.detach_()
        self.prev_c.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for t, xt in enumerate(xseq):
            assert xt.dim() == 4

            tmp = self.conv_h2h(self.prev_h) + xt

            cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            self.prev_c = f * self.prev_c + i * g
            self.prev_h = o * torch.tanh(self.prev_c)
            
            thr = torch.sigmoid(self.thr_reparam)
            event = SpikeFunction.apply(self.prev_h - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.prev_h - thr) < 0, thr-self.prev_h, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_h = self.prev_h * event
            result.append(self.prev_h)
            event_result.append(event)
            self.prev_h = self.prev_h - event * thr#.view(1,self.in_size[1],1,1)

        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
       
        if self.prev_h.numel() == 0:
            return
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask):
            assert batch_size == mask.numel()
            mask = mask.reshape(-1, 1, 1, 1)
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_c.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)
            self.prev_c = self.prev_c*mask.to(device=self.prev_c.device)

    @torch.jit.export
    def reset_all(self):
        """Resets memory for all sequences in one batch."""
        self.reset(torch.zeros((len(self.prev_h), 1, 1, 1), dtype=torch.float32, device=self.prev_h.device))

class ConvEventLSTMCell_ConditionalInput(nn.Module):
    
    def __init__(self, hidden_dim, kernel_size, in_size, conv_func=nn.Conv2d):
        super(ConvEventLSTMCell_ConditionalInput, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=True)
        self.conv_thr = conv_func(in_channels= 2*self.hidden_dim, out_channels= 1,
                                 kernel_size=kernel_size, padding=1)

        self.prev_h = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        self.prev_c = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        

    @torch.jit.export
    def get_dims_NCHW(self):
        return self.prev_h.size()

    def forward(self, x,stage):
        assert x.dim() == 5

        xseq = x.unbind(0)

        assert self.prev_h.size() == self.prev_c.size()

        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xseq[0].size()
        
        assert input_C == 4 * hidden_C
        assert hidden_C == self.hidden_dim

        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = x.device
            self.prev_h = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)
            self.prev_c = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)

        self.prev_h.detach_()
        self.prev_c.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for t, xt in enumerate(xseq):
            assert xt.dim() == 4

            tmp = self.conv_h2h(self.prev_h) + xt
            
            cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            self.prev_c = f * self.prev_c + i * g
            self.prev_h = o * torch.tanh(self.prev_c)

            thr = self.conv_thr(torch.cat((self.prev_h,self.prev_c), dim=1))
            thr = torch.sigmoid(thr)
            event = SpikeFunction.apply(self.prev_h - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.prev_h - thr) < 0, thr-self.prev_h, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_h = self.prev_h * event
            result.append(self.prev_h)
            event_result.append(event)
            self.prev_h = self.prev_h - event * thr#.view(1,self.in_size[1],1,1)

        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
       
        if self.prev_h.numel() == 0:
            return
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask):
            assert batch_size == mask.numel()
            mask = mask.reshape(-1, 1, 1, 1)
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_c.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)
            self.prev_c = self.prev_c*mask.to(device=self.prev_c.device)

    @torch.jit.export
    def reset_all(self):
        """Resets memory for all sequences in one batch."""
        self.reset(torch.zeros((len(self.prev_h), 1, 1, 1), dtype=torch.float32, device=self.prev_h.device))



class ConvEventMinGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.in_channels+self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            
            r = self.conv_rz(torch.cat((self.prev_y, xi), dim=1))
         
            forget_gate = torch.sigmoid(r)
            f = self.conv_f(torch.cat((self.prev_y * forget_gate, xi), dim=1))
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr = torch.sigmoid(self.thr_reparam)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventMinGRUCellFuseDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCellFuseDownsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [self.out_channels, self.out_channels], dim=1)
            r = self.conv_rz(self.prev_y) + xi_i
         
            forget_gate = torch.sigmoid(r)
            f = self.conv_f(self.prev_y * forget_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr = torch.sigmoid(self.thr_reparam)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)
            
class ConvEventMinGRUCellFuseDownsample_ConditionalHidden(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCellFuseDownsample_ConditionalHidden, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels+1,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        # self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
        #                                              math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [self.out_channels, self.out_channels], dim=1)
            r = self.conv_rz(self.prev_y) + xi_i
         
            forget_gate = torch.sigmoid(r)
            # f = self.conv_f(self.prev_y * forget_gate) + xi_f
            f, thr =self.conv_f(self.prev_y * forget_gate).split([self.out_channels, 1], dim = 1)
            f = f + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr = torch.sigmoid(thr)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventMinGRUCellFuseDownsample_ConditionalInput(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCellFuseDownsample_ConditionalInput, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels= self.out_channels+1,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        # self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
        #                                              math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [self.out_channels, self.out_channels], dim=1)
            # r = self.conv_rz(self.prev_y) + xi_i
            r, thr = self.conv_rz(self.prev_y).split([self.out_channels, 1], dim = 1)
            r = r + xi_i
            
            forget_gate = torch.sigmoid(r)
            f = self.conv_f(self.prev_y * forget_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr = torch.sigmoid(thr)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventMinGRUCellFuseDownsample_ConditionalDenseHidden(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCellFuseDownsample_ConditionalDenseHidden, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.conv_thr = conv_func(in_channels=self.out_channels, out_channels=1,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        # self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
        #                                              math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [self.out_channels, self.out_channels], dim=1)
            r = self.conv_rz(self.prev_y) + xi_i
         
            forget_gate = torch.sigmoid(r)
            f = self.conv_f(self.prev_y * forget_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr = self.conv_thr(self.dense_hidden_state)
            thr = torch.sigmoid(thr)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)
            
class ConvEventMinGRUCellFuseDownsample_get_thr_input(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinGRUCellFuseDownsample_get_thr_input, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt, thr):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi, thr_i in zip(xt,thr):
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [self.out_channels, self.out_channels], dim=1)
            r = self.conv_rz(self.prev_y) + xi_i
         
            forget_gate = torch.sigmoid(r)
            f = self.conv_f(self.prev_y * forget_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * input_gate
            thr_i = torch.sigmoid(thr_i)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr_i, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr_i) < 0, thr_i-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr_i  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)
            
class ConvEventMinRNN(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.in_channels+self.out_channels, out_channels= self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            z = torch.tanh(xi)
            r = self.conv_rz(torch.cat((self.prev_y, z), dim=1))
         
            forget_gate = torch.sigmoid(r)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * z
            thr = torch.sigmoid(self.thr_reparam)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventMinRNN_ConditionalInput(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventMinRNN_ConditionalInput, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=2*self.out_channels, out_channels= self.out_channels+1,
                                 kernel_size=kernel_size, padding=1)
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
    def forward(self, xt,stage):
      
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.dense_hidden_state.detach_()
        self.prev_y.detach_()

        result = []
        event_result = []
        negtive_distance_result = []
        for xi in xt:
            assert xi.dim() == 4
            z = torch.tanh(xi)
            r, thr= self.conv_rz(torch.cat((self.prev_y, z), dim=1)).split([self.out_channels, 1], dim = 1)
            
            forget_gate = torch.sigmoid(r)

            self.dense_hidden_state = (1 - forget_gate) * self.dense_hidden_state + forget_gate * z
            thr = torch.sigmoid(thr)
            
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            negtive_distance_result.append(negtive_distance)
            self.prev_y = self.dense_hidden_state * event
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr  
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, pruning = False,kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventGRUCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=2 * self.out_channels,
                                 kernel_size=kernel_size, padding=int(kernel_size/2))
        
        self.conv_f = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=int(kernel_size/2), stride=stride, dilation=dilation)
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        
        self.pruning = pruning
        if self.pruning:
            self.mask = []
            self.mask.append(torch.tensor(np.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/utils/dead_neuron_and_pruning/masks/mask128_egru0.npy')).to('cuda'))
            self.mask.append(torch.tensor(np.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/utils/dead_neuron_and_pruning/masks/mask128_egru1.npy')).to('cuda'))
            self.mask.append(torch.tensor(np.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/utils/dead_neuron_and_pruning/masks/mask128_egru2.npy')).to('cuda'))
        
    def forward(self, xt,stage):
        """
        xt size: (T, B,C,H,W)
        return size: (T, B,C',H,W)
        """
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.prev_y.detach_()
        self.dense_hidden_state.detach_()
        result = []
        event_result = []
        negtive_distance_result = []
        
        for t, xi in enumerate(xt):
            assert xi.dim() == 4

            z, r = self.conv_rz(torch.cat((self.prev_y, xi), dim=1)).split(self.out_channels, 1)
            update_gate = torch.sigmoid(z)
            reset_gate = torch.sigmoid(r)

            f = self.conv_f(torch.cat((self.prev_y * reset_gate, xi), dim=1))
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - update_gate) * self.dense_hidden_state + update_gate * input_gate
            
            thr = torch.sigmoid(self.thr_reparam)
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            if self.pruning:
                event = event * (self.mask[stage].view(1,256,1,1))
            
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            
            
            self.prev_y = self.dense_hidden_state * event #* torch.tensor(self.mask).view(1,256,1,1)
            negtive_distance_result.append(negtive_distance)
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr     #.view(1,self.in_size[1],1,1)
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventGRUCellFuseDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventGRUCellFuseDownsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels=2*self.out_channels,
                                 kernel_size=kernel_size, padding=int(kernel_size/2))
        
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=int(kernel_size/2), stride=stride, dilation=dilation)
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros(in_size) ,
                                                     math.sqrt(2) * torch.ones(in_size)))
        
        
    def forward(self, xt,stage):
        """
        xt size: (T, B,C,H,W)
        return size: (T, B,C',H,W)
        """
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.prev_y.detach_()
        self.dense_hidden_state.detach_()
        result = []
        event_result = []
        negtive_distance_result = []
        
        for t, xi in enumerate(xt):
            assert xi.dim() == 4
            xi_i, xi_f = torch.split(xi, [2*self.out_channels, self.out_channels], dim=1)
            
            z, r = (self.conv_rz(self.prev_y) + xi_i).split(self.out_channels, 1)
            update_gate = torch.sigmoid(z)
            reset_gate = torch.sigmoid(r)

            f = self.conv_f(self.prev_y * reset_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - update_gate) * self.dense_hidden_state + update_gate * input_gate
            
            thr = torch.sigmoid(self.thr_reparam)
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            
            
            self.prev_y = self.dense_hidden_state * event #* torch.tensor(self.mask).view(1,256,1,1)
            negtive_distance_result.append(negtive_distance)
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr     #.view(1,self.in_size[1],1,1)
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)

class ConvEventGRUCellFuseDownsample_ConditionalInput(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size=3, padding=1, conv_func=nn.Conv2d, 
                 stride=1, dilation=1):
        super(ConvEventGRUCellFuseDownsample_ConditionalInput, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.out_channels, out_channels=2*self.out_channels+1,
                                 kernel_size=kernel_size, padding=int(kernel_size/2))
        
        self.conv_f = conv_func(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=int(kernel_size/2), stride=stride, dilation=dilation)
        self.prev_y = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        self.dense_hidden_state = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)
        
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.in_size = in_size
        
        
        
    def forward(self, xt,stage):
        """
        xt size: (T, B,C,H,W)
        return size: (T, B,C',H,W)
        """
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_y.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_y = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)
            self.dense_hidden_state = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,device=device)
            
        self.prev_y.detach_()
        self.dense_hidden_state.detach_()
        result = []
        event_result = []
        negtive_distance_result = []
        
        for t, xi in enumerate(xt):
            assert xi.dim() == 4
            xi_i_z, xi_i_r, xi_f = torch.split(xi, [self.out_channels, self.out_channels, self.out_channels], dim=1)
            
            z, r, thr = self.conv_rz(self.prev_y).split([self.out_channels, self.out_channels, 1], dim = 1)
            z = z + xi_i_z
            r = r + xi_i_r
            update_gate = torch.sigmoid(z)
            reset_gate = torch.sigmoid(r)

            f = self.conv_f(self.prev_y * reset_gate) + xi_f
            input_gate = torch.tanh(f)

            self.dense_hidden_state = (1 - update_gate) * self.dense_hidden_state + update_gate * input_gate
            
            thr = torch.sigmoid(thr)
            event = SpikeFunction.apply(self.dense_hidden_state - thr, self.dampening_factor, self.pseudo_derivative_support)
            
            negtive_distance = torch.where((self.dense_hidden_state - thr) < 0, thr-self.dense_hidden_state, 0)
            
            
            self.prev_y = self.dense_hidden_state * event #* torch.tensor(self.mask).view(1,256,1,1)
            negtive_distance_result.append(negtive_distance)
            result.append(self.prev_y)
            event_result.append(event)
            self.dense_hidden_state = self.dense_hidden_state - event * thr     #.view(1,self.in_size[1],1,1)
            
        return torch.cat([r[None] for r in result], dim=0), torch.cat([r[None] for r in event_result], dim=0), torch.cat([r[None] for r in negtive_distance_result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        batch_size, _, _, _ = self.prev_y.size()
        if batch_size == len(mask) and self.prev_y.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_y), 1, 1, 1])
            self.prev_y.detach_()
            self.prev_y = self.prev_y*mask.to(device=self.prev_y.device)




class ConvRNN(nn.Module):
    def __init__(self, in_channels, out_channels, in_size = None, pruning = False, stage = 0, kernel_size=3, stride=1, padding=1, dilation=1,
                 cell='lstm', separable=False, separable_hidden=False, **kwargs):
        super(ConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.cell = cell
        self.get_thr_from_input = False
        
        if cell.lower() == "elstm":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 4 * out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventLSTMCell(out_channels, 3, in_size = in_size, conv_func=nn.Conv2d)
        
        elif cell.lower() == "elstmconditionalinput":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 4 * out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventLSTMCell_ConditionalInput(out_channels, 3, in_size = in_size, conv_func=nn.Conv2d)
        
        elif cell.lower() == "egrurelu":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, in_channels, kernel_size=kernel_size,
                      activation='ReLU', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventGRUCell(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size, pruning=pruning)
        
        elif cell.lower() == "egrufusedownsample":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 3*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventGRUCellFuseDownsample(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
            
        elif cell.lower() == "egrufusedownsampleconditionalinput":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 3*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventGRUCellFuseDownsample_ConditionalInput(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
        
        elif cell.lower() == "egrupruning":
            if stage == 0:
                self.conv_x2h = SequenceWise(
                ConvLayer(in_channels, in_channels, kernel_size=kernel_size,
                        activation='ReLU', stride=stride, padding=padding, dilation=dilation,
                        **kwargs))
            else:
                self.conv_x2h = SequenceWise(
                ConvLayer(in_channels, 256, kernel_size=kernel_size,
                        activation='ReLU', stride=stride, padding=padding, dilation=dilation,
                        **kwargs))
            self.timepool = ConvEventGRUCell(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size, pruning=pruning)
        
        elif cell.lower() == "emgru":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, in_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCell(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)

        elif cell.lower() == "emgrufusedownsample":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 2*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCellFuseDownsample(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
        
        elif cell.lower() == "emgrufusedownsampleconditionalhidden":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 2*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCellFuseDownsample_ConditionalHidden(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
        
        elif cell.lower() == "emgrufusedownsampleconditionalinput":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 2*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCellFuseDownsample_ConditionalInput(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
            
        elif cell.lower() == "conditionalconvemgrufusedownsampleconditionalinput":
            self.conv_x2h = SequenceWise(
            ConditionalConvLayer(in_channels, 2*out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding))
            self.timepool = ConvEventMinGRUCellFuseDownsample_ConditionalInput(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)    
        
        elif cell.lower() == "emgrufusedownsampleconditionaldensehidden":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 2*out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCellFuseDownsample_ConditionalDenseHidden(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
            
        elif cell.lower() == "emgrufusedownsamplegetthrinput":
            self.get_thr_from_input = True
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 2*out_channels+1, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinGRUCellFuseDownsample_get_thr_input(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
        
        elif cell.lower() == "eminrnnrelu":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, out_channels, kernel_size=kernel_size,
                      activation='ReLU', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinRNN(out_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
            
        elif cell.lower() == "eminrnnconditionalinput":
            self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, out_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation,
                      **kwargs))
            self.timepool = ConvEventMinRNN_ConditionalInput(out_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d, in_size = in_size)
            
        else:
            raise NotImplementedError()

    def forward(self, x, stage = 0):
        if not self.get_thr_from_input:
            if self.cell.lower() == "egrurelu" or self.cell.lower() == "egruidentity" or self.cell.lower() == "eminrnnrelu" or self.cell.lower() == "elstm" or self.cell.lower() == "emgru" or self.cell.lower() == 'egrufusedownsample' or self.cell.lower() ==  'emgrufusedownsample' or self.cell.lower() == 'emgrufusedownsampleconditionalhidden' or self.cell.lower() == 'emgrufusedownsampleconditionalinput' or self.cell.lower() == 'emgrufusedownsampleconditionaldensehidden' or self.cell.lower() == "elstmconditionalinput" or self.cell.lower() == "eminrnnconditionalinput" or self.cell.lower() == "egrufusedownsampleconditionalinput" or self.cell.lower() == "conditionalconvemgrufusedownsampleconditionalinput": 
                y = self.conv_x2h(x)
                h, events, neg_dis = self.timepool(y,stage)
                return h, events, neg_dis, y
        
        else:
            if self.cell.lower() == "emgrufusedownsamplegetthrinput":
                y, thr = self.conv_x2h(x).split([2*self.out_channels, 1], dim = 2)
                h, events, neg_dis = self.timepool(y, thr)
                return h, events, neg_dis, y
            
     
    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        """Resets memory of the network."""
        self.timepool.reset(mask)
    
    def reset_half(self, mask=torch.zeros((1,), dtype=torch.float32)):
        """Resets memory of the network."""
        self.timepool.reset_half(mask)

    @torch.jit.export
    def reset_all(self):
        self.timepool.reset_all()



