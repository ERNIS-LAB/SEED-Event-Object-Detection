# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Box Regression and Classification
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Module.basic_module import ConvLayer
from typing import List


def softmax_init(l, num_anchors, num_logits):
    """
    softmax initialization:

    We derive initialization according to target probability of background 0.99
    Focal Loss for Dense Object Detection (Lin et al.)

    Args:
        l: linear layer
        num_anchors: number of anchors of prediction
        num_logits: number of classes + 1
    """
    px = 0.99
    num_classes = num_logits - 1
    bias_bg = math.log(num_classes * px / (1 - px))
    torch.nn.init.normal_(l.weight, std=0.01)
    torch.nn.init.constant_(l.bias, 0)
    l.bias.data = l.bias.data.reshape(num_anchors, num_logits)
    l.bias.data[:, 0] += bias_bg
    l.bias.data = l.bias.data.reshape(-1)


class BoxHead(nn.Module):
    """ Shared Prediction for boxes.
        Applies 2 small mini-convnets of stride1 to predict class and box delta.
        Reshape the predictions and concatenate them to output 2 vectors "loc" and "cls"
    """

    def __init__(self, in_channels, num_anchors, num_logits, n_layers=3):
        super(BoxHead, self).__init__()
        self.num_logits = num_logits
        self.in_channels = in_channels
        self.num_anchors = num_anchors

        self.aspect_ratios = []

        def conv_func(x, y): return ConvLayer(x, y, norm='none', activation='ReLU')
        self.box_head = self._make_head(in_channels, self.num_anchors * 4, n_layers, conv_func)
        self.cls_head = self._make_head(in_channels, self.num_anchors * self.num_logits, n_layers, conv_func)

        def initialize_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.cls_head.apply(initialize_layer)
        self.box_head.apply(initialize_layer)

        softmax_init(self.cls_head[-1], self.num_anchors, self.num_logits)

    def init_cls_head(self):
        softmax_init(self.cls_head[-1], self.num_anchors, self.num_logits)

    def _make_head(self, in_planes, out_planes, n_layers, conv_func):
        layers = []
        layers.append(conv_func(in_planes, 256))
        for _ in range(n_layers):
            layers.append(conv_func(256, 256))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def _apply_head(self, layer, xs, ndims):
        out = []
        for x in xs:
            y = layer(x).permute(0, 2, 3, 1).contiguous()
            y = y.view(y.size(0), -1, ndims)
            out.append(y)
        out = torch.cat(out, 1)
        return out

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        loc_preds_list = []
        cls_preds_list = []
        box_dims = 4
        for x in xs:
            y_box = self.box_head(x).permute(0, 2, 3, 1).contiguous()
            y_box = y_box.view(y_box.size(0), -1, box_dims)
            loc_preds_list.append(y_box)

            y_cls = self.cls_head(x).permute(0, 2, 3, 1).contiguous()
            y_cls = y_cls.view(y_cls.size(0), -1, self.num_logits)
            cls_preds_list.append(y_cls)

        loc_preds = torch.cat(loc_preds_list, 1)
        cls_preds = torch.cat(cls_preds_list, 1)
        return [loc_preds, cls_preds, [self.box_head[0](x).detach_() for x in xs], [self.cls_head[0](x).detach_() for x in xs] ]

    def probas(self, cls_preds):
        return F.softmax(cls_preds, dim=2)

    @torch.jit.export
    def get_scores(self, cls_preds):
        cls = self.probas(cls_preds)
        cls = cls[..., 1:].contiguous()  # first logits is for class "background"
        return cls

    def get_classification_params(self):
        # reshape into num_anchors, num_logits
        w = self.cls_head[-1].weight
        w = w.reshape(self.num_anchors, self.num_logits, *w.shape[1:])
        b = self.cls_head[-1].bias
        b = b.reshape(self.num_anchors, self.num_logits)
        return w, b
