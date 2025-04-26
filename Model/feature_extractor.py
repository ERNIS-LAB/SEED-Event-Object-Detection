import numpy as np
import torch
import torch.nn as nn



from Module.basic_module import ConvLayer, EventConvLayer, ResBlockSplit, EventResBlock, ConditionalEventResBlock, ConditionalConvLayer
from Module.temporal_module import time_to_batch, SequenceWise, ConvRNN,batch_to_time



# class ELSTMReLu_resnet_sparse(nn.Module):

#     def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
#         super(ELSTMReLu_resnet_sparse, self).__init__()
#         self.cin = cin
#         self.cout = cout
#         self.base = base
#         self.levels = 3
        
#         self.conv1 = nn.ModuleList()
#         self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
#         self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
#         if dataset == 'gen4' and head_type == 'ssd':
#             size_layer0 = [cout,45,80]
#             size_layer1 = [cout,23,40]
#             size_layer2 = [cout,12,20]
#         elif dataset == 'gen1' and head_type == 'ssd':
#             size_layer0 = [cout,30,38]
#             size_layer1 = [cout,15,19]
#             size_layer2 = [cout,8,10]
#         elif dataset == 'gen4' and head_type == 'yolo':
#             size_layer0 = [cout,48,80]
#             size_layer1 = [cout,24,40]
#             size_layer2 = [cout,12,20]
        
#         self.conv2 = nn.ModuleList()
#         self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="elstm", in_size = size_layer0, pruning = pruning))
        
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstm", in_size = size_layer1, pruning = pruning))
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstm", in_size = size_layer2, pruning = pruning))
        
#     def forward(self, x):
#         conv1_out = []

        
#         y4, batch_size = time_to_batch(x)
        
#         y4 = self.conv1[0](y4)
#         conv1_out.append(y4)
        
#         for conv in self.conv1[1:]:
#             resout, y4 = conv(y4)
#             conv1_out.append(resout)
#             conv1_out.append(y4)
            
#         x = batch_to_time(y4, batch_size)
        
#         outs = []
#         events_out = []
#         neg_dis_out = []
#         reluouts = []
#         for stage, conv in enumerate(self.conv2):
#             x, events, neg_dis, reluout = conv(x,stage)
#             y = time_to_batch(x)[0]
#             reluout = time_to_batch(reluout)[0]
#             outs.append(y)
#             reluouts.append(reluout)
#             events_out.append(events)
#             neg_dis_out.append(neg_dis)
#         return outs, events_out, neg_dis_out, conv1_out, reluouts

#     def reset(self, mask=None):
#         for name, module in self.conv2.named_modules():
#             if hasattr(module, "reset"):
#                 module.reset(mask)

#     @torch.jit.export
#     def reset_all(self):
#         for module in self.conv2:
#             module.reset_all()
            
# class EMinRNN_ReLu(nn.Module):
#     def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4'):
#         super(EMinRNN_ReLu, self).__init__()
#         self.cin = cin
#         self.cout = cout
#         self.base = base
#         self.levels = 3

#         self.conv1 = nn.ModuleList()
#         self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
#         self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
#         size_layer0 = [256,45,80]
#         size_layer1 = [256,23,40]
#         size_layer2 = [256,12,20]
#         # size_layer3 = [256,6,10]
#         # size_layer4 = [256,3,5]
        

#         self.conv2 = nn.ModuleList()
#         self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="eminrnnrelu", in_size = size_layer0))
        
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnrelu", in_size = size_layer1))
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnrelu", in_size = size_layer2))
       
#     def forward(self, x):
#         conv1_out = []

        
#         y4, batch_size = time_to_batch(x)
        
#         y4 = self.conv1[0](y4)
#         conv1_out.append(y4)
        
#         for conv in self.conv1[1:]:
#             resout, y4 = conv(y4)
#             conv1_out.append(resout)
#             conv1_out.append(y4)
            
#         x = batch_to_time(y4, batch_size)
        
#         outs = []
#         events_out = []
#         neg_dis_out = []
#         reluouts = []
#         for stage, conv in enumerate(self.conv2):
#             x, events, neg_dis, reluout = conv(x,stage)
#             y = time_to_batch(x)[0]
#             reluout = time_to_batch(reluout)[0]
#             outs.append(y)
#             reluouts.append(reluout)
#             events_out.append(events)
#             neg_dis_out.append(neg_dis)
#         return outs, events_out, neg_dis_out, conv1_out, reluouts

#     def reset(self, mask=None):
#         for name, module in self.conv2.named_modules():
#             if hasattr(module, "reset"):
#                 module.reset(mask)

#     @torch.jit.export
#     def reset_all(self):
#         for module in self.conv2:
#             module.reset_all()
   
# class EMinGRU_ReLU(nn.Module):

#     def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
#         super(EMinGRU_ReLU, self).__init__()
#         self.cin = cin
#         self.cout = cout
#         self.base = base
#         self.levels = 3
        
#         self.conv1 = nn.ModuleList()
#         self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
#         self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
#         self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
#         if dataset == 'gen4' and head_type == 'ssd':
#             size_layer0 = [cout,45,80]
#             size_layer1 = [cout,23,40]
#             size_layer2 = [cout,12,20]
#         elif dataset == 'gen1' and head_type == 'ssd':
#             size_layer0 = [cout,30,38]
#             size_layer1 = [cout,15,19]
#             size_layer2 = [cout,8,10]
#         elif dataset == 'gen4' and head_type == 'yolo':
#             size_layer0 = [cout,48,80]
#             size_layer1 = [cout,24,40]
#             size_layer2 = [cout,12,20]
        
#         self.conv2 = nn.ModuleList()
#         self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgru", in_size = size_layer0, pruning = pruning))
        
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgru", in_size = size_layer1, pruning = pruning))
#         self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgru", in_size = size_layer2, pruning = pruning))
        
#     def forward(self, x):
#         conv1_out = []

        
#         y4, batch_size = time_to_batch(x)
        
#         y4 = self.conv1[0](y4)
#         conv1_out.append(y4)
        
#         for conv in self.conv1[1:]:
#             resout, y4 = conv(y4)
#             conv1_out.append(resout)
#             conv1_out.append(y4)
            
#         x = batch_to_time(y4, batch_size)
        
#         outs = []
#         events_out = []
#         neg_dis_out = []
#         reluouts = []
#         for stage, conv in enumerate(self.conv2):
#             x, events, neg_dis, reluout = conv(x,stage)
#             y = time_to_batch(x)[0]
#             reluout = time_to_batch(reluout)[0]
#             outs.append(y)
#             reluouts.append(reluout)
#             events_out.append(events)
#             neg_dis_out.append(neg_dis)
#         return outs, events_out, neg_dis_out, conv1_out, reluouts

#     def reset(self, mask=None):
#         for name, module in self.conv2.named_modules():
#             if hasattr(module, "reset"):
#                 module.reset(mask)

#     @torch.jit.export
#     def reset_all(self):
#         for module in self.conv2:
#             module.reset_all()           
   
class ELSTM_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ELSTM_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="elstm", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstm", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstm", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
 
class ConditionalInput_ELSTM_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalInput_ELSTM_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="elstmconditionalinput", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstmconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="elstmconditionalinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
        
            
class EMinRNN_ReLu_ConditionalConv(nn.Module):
    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', pruning = False, threshold_group = 1):
        super(EMinRNN_ReLu_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3

        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        size_layer0 = [256,45,80]
        size_layer1 = [256,23,40]
        size_layer2 = [256,12,20]
        # size_layer3 = [256,6,10]
        # size_layer4 = [256,3,5]
        

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="eminrnnrelu", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnrelu", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnrelu", in_size = size_layer2, pruning = pruning))
       
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
            
class ConditionalInput_EMinRNN_ConditionalConv(nn.Module):
    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', pruning = False, threshold_group = 1):
        super(ConditionalInput_EMinRNN_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3

        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        size_layer0 = [256,45,80]
        size_layer1 = [256,23,40]
        size_layer2 = [256,12,20]
        # size_layer3 = [256,6,10]
        # size_layer4 = [256,3,5]
        

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="eminrnnconditionalinput", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="eminrnnconditionalinput", in_size = size_layer2, pruning = pruning))
       
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
   
   
class EMinGRU_ReLUFuseDownsampleConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(EMinGRU_ReLUFuseDownsampleConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()             
   
class EMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(EMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()           
 
class ConditionalHidden_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalHidden_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsampleconditionalhidden", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalhidden", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalhidden", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()           
 
class ConditionalInput_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalInput_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()    
            
class ConditionalInput_ConditionalconvEMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalInput_ConditionalconvEMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="conditionalconvemgrufusedownsampleconditionalinput", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="conditionalconvemgrufusedownsampleconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="conditionalconvemgrufusedownsampleconditionalinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()   
            
class ConditionalDenseHidden_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalDenseHidden_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsampleconditionaldensehidden", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionaldensehidden", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionaldensehidden", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()     
            
class ConditionalGetThrInput_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalGetThrInput_EMinGRU_ReLUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsamplegetthrinput", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsamplegetthrinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsamplegetthrinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()                 
 
class ConditionalInput_EMinGRU_ReLUFuseDownsampleConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalInput_EMinGRU_ReLUFuseDownsampleConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsampleconditionalinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()                         

class EMinGRU_ReLUFuseDownsampleConv_EventConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(EMinGRU_ReLUFuseDownsampleConv_EventConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
            in_height = [180, 90, 90, 90]
            in_width = [320, 160, 160, 160]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
            in_height = [120, 60, 60, 60]
            in_width = [152, 76, 76, 76]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(EventConvLayer(cin, self.base * 2, in_height[0], in_width[0], kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(EventResBlock(self.base * 2, self.base * 4, in_height[1], in_width[1], 2))
        self.conv1.append(EventResBlock(self.base * 4, self.base * 4, in_height[2], in_width[2], 1))
        self.conv1.append(EventResBlock(self.base * 4, self.base * 8, in_height[3], in_width[3], 1))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="emgrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            outs.append(y)
            reluouts.append(reluout)
            events_out.append(events)
            neg_dis_out.append(neg_dis)
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()        
            
class SEED_EventGRU(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
        super(SEED_EventGRU, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="egruRelu", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egruRelu", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egruRelu", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()

class SEED_EventGRUFuseDownsampleConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
        super(SEED_EventGRUFuseDownsampleConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(ResBlockSplit(self.base * 2, self.base * 4, 2))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 4, 1))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 1))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="egrufusedownsample", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()

class SEED_EventGRUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(SEED_EventGRUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
            in_height = [180, 90, 90, 90]
            in_width = [320, 160, 160, 160]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="egrufusedownsample", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
      
class ConditionalInput_EventGRUFuseDownsampleConv_ConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalInput_EventGRUFuseDownsampleConv_ConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
            in_height = [180, 90, 90, 90]
            in_width = [320, 160, 160, 160]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="egrufusedownsampleconditionalinput", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsampleconditionalinput", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsampleconditionalinput", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
      
class SEED_EventGRUFuseDownsampleConv_DoubleConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
        super(SEED_EventGRUFuseDownsampleConv_DoubleConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConvLayer(cin, self.base * 4, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(ResBlockSplit(self.base * 4, self.base * 8, 2))
        self.conv1.append(ResBlockSplit(self.base * 8, self.base * 8, 1))
        self.conv1.append(ResBlockSplit(self.base * 8, self.base * 16, 1))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 16, cout, stride=2, cell="egrufusedownsample", in_size = size_layer0, pruning = pruning))
        
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
    
    
class SEED_EventGRUFuseDownsampleConv_DoubleEventConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False):
        super(SEED_EventGRUFuseDownsampleConv_DoubleEventConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
            in_height = [180, 90, 90, 90]
            in_width = [320, 160, 160, 160]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(EventConvLayer(cin, self.base * 4, in_height[0], in_width[0], kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'))
        self.conv1.append(EventResBlock(self.base * 4, self.base * 8, in_height[1], in_width[1], 2))
        self.conv1.append(EventResBlock(self.base * 8, self.base * 8, in_height[2], in_width[2], 1))
        self.conv1.append(EventResBlock(self.base * 8, self.base * 16, in_height[3], in_width[3], 1))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 16, cout, stride=2, cell="egrufusedownsample", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
      
      
class SEED_EventGRUFuseDownsampleConv_DoubleConditionalConv(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(SEED_EventGRUFuseDownsampleConv_DoubleConditionalConv, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
            in_height = [180, 90, 90, 90]
            in_width = [320, 160, 160, 160]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 4, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 8, self.base * 8, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 8, self.base * 16, 1, threshold_group=threshold_group))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 16, cout, stride=2, cell="egrufusedownsample", in_size = size_layer0, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer1, pruning = pruning))
        self.conv2.append(ConvRNN(cout, cout, stride=2, cell="egrufusedownsample", in_size = size_layer2, pruning = pruning))
        
    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x = batch_to_time(y4, batch_size)
        
        outs = []
        events_out = []
        neg_dis_out = []
        reluouts = []
        for stage, conv in enumerate(self.conv2):
            x, events, neg_dis, reluout = conv(x,stage)
            
            y = time_to_batch(x)[0]
            reluout = time_to_batch(reluout)[0]
            reluouts.append(reluout)
            events_out.append(time_to_batch(events)[0])
            
            outs.append(y)
            neg_dis_out.append(time_to_batch(neg_dis)[0])
        return outs, events_out, neg_dis_out, conv1_out, reluouts

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
      
class ConditionalConv_convNN(nn.Module):

    def __init__(self, cin=1, base=16, cout=256, dataset = 'gen4', head_type = 'ssd', pruning = False, threshold_group = 1):
        super(ConditionalConv_convNN, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 3
        
        self.conv1 = nn.ModuleList()
        self.conv1.append(ConditionalConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, threshold_group=threshold_group, norm='BatchNorm2d'))
        self.conv1.append(ConditionalEventResBlock(self.base * 2, self.base * 4, 2, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 4, 1, threshold_group=threshold_group))
        self.conv1.append(ConditionalEventResBlock(self.base * 4, self.base * 8, 1, threshold_group=threshold_group))
        
        if dataset == 'gen4' and head_type == 'ssd':
            size_layer0 = [cout,45,80]
            size_layer1 = [cout,23,40]
            size_layer2 = [cout,12,20]
        elif dataset == 'gen1' and head_type == 'ssd':
            size_layer0 = [cout,30,38]
            size_layer1 = [cout,15,19]
            size_layer2 = [cout,8,10]
        elif dataset == 'gen4' and head_type == 'yolo':
            size_layer0 = [cout,48,80]
            size_layer1 = [cout,24,40]
            size_layer2 = [cout,12,20]

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvLayer(self.base * 8, cout, kernel_size=3, stride=2, padding=1, norm='BatchNorm2d'))
        
        for i in range(self.levels - 1):
            self.conv2.append(ConvLayer(cout, cout, kernel_size=3, stride=2, padding=1, norm='BatchNorm2d'))

    def forward(self, x):
        conv1_out = []

        
        y4, batch_size = time_to_batch(x)
        
        y4 = self.conv1[0](y4)
        conv1_out.append(y4)
        
        for conv in self.conv1[1:]:
            resout, y4 = conv(y4)
            conv1_out.append(resout)
            conv1_out.append(y4)
            
        x4 = y4
        outs = []
        for conv in self.conv2:
            x4 = conv(x4)
            # y = time_to_batch(x)[0]
            outs.append(x4)
        return outs, conv1_out

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()