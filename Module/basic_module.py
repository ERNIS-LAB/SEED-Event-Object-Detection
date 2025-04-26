
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .spiking import SpikeFunction

class ConvLayer(nn.Sequential):
   
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, norm="BatchNorm2d", activation='ReLU'):

        conv_func = nn.Conv2d
        self.out_channels = out_channels

        normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels)

        super(ConvLayer, self).__init__(
            conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=padding, bias=bias),
            normalizer,
            getattr(nn, activation)()
        )
        
        
class ResBlockSplit(nn.Module):
    

    def __init__(self, in_channels, out_channels, stride=1, norm="BatchNorm2d"):
        super(ResBlockSplit, self).__init__()
        bias = norm == 'none'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm=norm,
        )
        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm=norm,
            bias=False,
        )

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                norm=norm,
                bias=False,
                activation="Identity",
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        out = out + self.downsample(x)
        out = F.relu(out)
        return out1, out


class EventConvLayer(nn.Module):
   
    def __init__(self, in_channels, out_channels, height, width,
                 kernel_size=3, stride=1, padding=1, bias=True, norm="BatchNorm2d"):
        super(EventConvLayer, self).__init__()
        self.out_channels = out_channels
        conv2d_NoActive = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels)
        self.conv = nn.Sequential(conv2d_NoActive, normalizer)
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        self.thr_reparam = nn.Parameter(torch.normal(torch.zeros([height, width]) ,
                                                     math.sqrt(2) * torch.ones([height, width])))
        
    def forward(self, x):
        x = self.conv(x)
        thr = torch.sigmoid(self.thr_reparam)
        event = SpikeFunction.apply(x - thr, self.dampening_factor, self.pseudo_derivative_support)
        outs = x * event
        return outs
    
class EventResBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels, height, width, stride=1, norm="BatchNorm2d"):
        super(EventResBlock, self).__init__()
        bias = norm == 'none'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = EventConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            height=height*stride,
            width=width*stride,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm=norm,
        )
        self.conv2 = EventConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            height=height,
            width=width,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm=norm,
            bias=False,
        )

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = EventConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                height=height,
                width=width,
                kernel_size=1,
                padding=0,
                stride=stride,
                norm=norm,
                bias=False,
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        out = out + self.downsample(x)
        out = F.relu(out)
        return out1, out


class ConditionalConvLayer(nn.Module):
   
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=True, threshold_group=1, norm="BatchNorm2d"):
        super(ConditionalConvLayer, self).__init__()
        self.out_channels = out_channels
        self.threshold_group = threshold_group
        self.conv2d_NoActive = nn.Conv2d(in_channels, out_channels + threshold_group, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels)
        # self.conv = nn.Sequential(conv2d_NoActive, normalizer)
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        
    def forward(self, x):
        x, thr = self.conv2d_NoActive(x).split([self.out_channels, self.threshold_group], dim = 1)
        x = self.normalizer(x)
        thr = torch.sigmoid(thr)
        x_split = x.split(self.out_channels//self.threshold_group, dim = 1)
        thr_split = thr.split(1, dim = 1)
        result = torch.cat([x_part - thr_part for x_part, thr_part in zip(x_split, thr_split)], dim=1)
        event = SpikeFunction.apply(result, self.dampening_factor, self.pseudo_derivative_support)
        outs = x * event
        return outs
    
class ConditionalEventResBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels, stride=1, threshold_group=1, norm="BatchNorm2d"):
        super(ConditionalEventResBlock, self).__init__()
        bias = norm == 'none'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.threshold_group = threshold_group
        self.conv1 = ConditionalConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            threshold_group=threshold_group,
            norm=norm,
        )
        
        self.conv2 = nn.Conv2d(out_channels, out_channels + threshold_group, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                norm=norm,
                bias=False,
                activation="Identity"
            )
            
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.9]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)
        

    def forward(self, x):
        out1 = self.conv1(x)
        out,thr = self.conv2(out1).split([self.out_channels, self.threshold_group], dim = 1)
        out = self.conv2_normalizer(out)
        out = out + self.downsample(x)
        thr = torch.sigmoid(thr)
        out_split = out.split(self.out_channels//self.threshold_group, dim = 1)
        thr_split = thr.split(1, dim = 1)
        result = torch.cat([out_part - thr_part for out_part, thr_part in zip(out_split, thr_split)], dim=1)
        event = SpikeFunction.apply(result, self.dampening_factor, self.pseudo_derivative_support)
        out = out * event
        return out1, out