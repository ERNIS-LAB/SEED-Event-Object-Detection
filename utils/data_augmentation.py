import torch
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
import numpy as np
import pandas as pd

def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * torch.rand(1).item()

class data_augmentation: # zoom out and flip
    def __init__(self, dataset_type = 'gen4') -> None:
        if dataset_type == 'gen4':
            self.height = 360
            self.width = 640
        else:
            self.height = 240
            self.width = 304
        
        self.h_flip_prob = 0.5
        self.zoom_prob = 0.6
        

        self.h_flip_state = False
        self.zoom_state = False
        self.zoom_out_state = False
        self.zoom_in_state = False
        
        self.min_zoom_out_factor = 1
        self.max_zoom_out_factor = 1.2
        self.zoom_out_factor = 0
        self.zoom_out_x0 = 0
        self.zoom_out_y0 = 0
        
        self.min_zoom_in_factor = 1
        self.max_zoom_in_factor = 1.2
        self.zoom_in_factor = 0
        self.zoom_in_x0 = 0
        self.zoom_in_y0 = 0
        
        
        zoom_in_weight = 3.5
        zoom_out_weight = 6.5
        self.zoom_in_or_out_distribution = torch.distributions.categorical.Categorical(
            probs=torch.tensor([zoom_in_weight, zoom_out_weight]))
        
        
    def randomize_augmentation(self):
       
        self.h_flip_state = self.h_flip_prob > torch.rand(1).item()
        
        self.zoom_state = self.zoom_prob > torch.rand(1).item()
        self.zoom_in_state = self.zoom_in_or_out_distribution.sample().item() == 0
        self.zoom_out_state = not self.zoom_in_state
        self.zoom_in_state &= self.zoom_state
        self.zoom_out_state &= self.zoom_state
        
        if self.zoom_out_state:
            rand_zoom_out_factor = torch_uniform_sample_scalar(
                min_value=self.min_zoom_out_factor, max_value=self.max_zoom_out_factor)
            zoom_window_h, zoom_window_w = int(self.height / rand_zoom_out_factor), int(self.width / rand_zoom_out_factor)
            x0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=self.width - zoom_window_w))
            y0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=self.height - zoom_window_h))
            self.zoom_out_x0 = x0_sampled
            self.zoom_out_y0 = y0_sampled
            self.zoom_out_factor = rand_zoom_out_factor
            
        if self.zoom_in_state:
            rand_zoom_in_factor = torch_uniform_sample_scalar(
                min_value=self.min_zoom_in_factor, max_value=self.max_zoom_in_factor)
            zoom_window_h, zoom_window_w = int(self.height / rand_zoom_in_factor), int(self.width / rand_zoom_in_factor)
            x0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=self.width - zoom_window_w))
            y0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=self.height - zoom_window_h))
            self.zoom_in_x0 = x0_sampled
            self.zoom_in_y0 = y0_sampled
            self.zoom_in_factor = rand_zoom_in_factor
    
    def __call__(self, data_dict, only_vertical_move=False):
        if not only_vertical_move:
            self.randomize_augmentation()

            if self.h_flip_state:
                data_dict = self.h_flip(data_dict)
                
            if self.zoom_out_state:
                data_dict = self.zoom_out(data_dict)
                
            if self.zoom_in_state:
                data_dict = self.zoom_in(data_dict)
            
            return data_dict
        
        else:
            return self.vertical_move(data_dict)
    
    def h_flip(self, data_dict):
        # data:
        
        data_dict['inputs'] = torch.flip(data_dict['inputs'][:][:], dims=[-1])
        # data_dict['inputs'] = [[torch.flip(tensor, dims=[-1]) for tensor in sublist] for sublist in data_dict['inputs']]
        
        # label:
        for i in range(len(data_dict["labels"])): #tbins
            for j in range(len(data_dict["labels"][i])):
                    labels = data_dict["labels"][i][j]
                    if isinstance(labels, np.ndarray):
                        labels['x'] = self.width - labels['x'] - labels['w']
                        data_dict["labels"][i][j] = labels

        return data_dict
    
    def vertical_move(self, data_dict):
        # data:
        mid_height = self.height//2
        tmp = data_dict['inputs'][:, :, :, :mid_height, :]
        data_dict['inputs'][:, :, :, :mid_height, :] = data_dict['inputs'][:, :, :, mid_height:, :]
        data_dict['inputs'][:, :, :, mid_height:, :] = tmp
        # data_dict['inputs'] = [[torch.flip(tensor, dims=[-1]) for tensor in sublist] for sublist in data_dict['inputs']]
        
        # label:
        for i in range(len(data_dict["labels"])): #tbins
            for j in range(len(data_dict["labels"][i])):
                    labels = data_dict["labels"][i][j]
                    if isinstance(labels, np.ndarray):
                        
                        labels['y'] = np.where(labels['y']>mid_height, labels['y']-mid_height ,0)
                 
                        data_dict["labels"][i][j] = labels[labels['y']!=0]
                       

        return data_dict
    
    
    def zoom_out(self, data_dict): #suoxiao
        zoom_window_h, zoom_window_w = int(self.height / self.zoom_out_factor), int(self.width / self.zoom_out_factor)
        
        t, n, c, h, w = data_dict['inputs'].size()
        x = data_dict['inputs'].view(n * t, c, h, w)
       
        zoom_window = interpolate(x, size=(zoom_window_h, zoom_window_w), mode='nearest-exact')
        zoom_window = zoom_window.view(t, n, c, zoom_window_h,zoom_window_w ).contiguous()
        
        output = torch.zeros_like(data_dict['inputs'])
        
        output[..., self.zoom_out_y0:self.zoom_out_y0 + zoom_window_h, self.zoom_out_x0:self.zoom_out_x0 + zoom_window_w] = zoom_window
        data_dict['inputs'] = output
        
        
        # label:
        for i in range(len(data_dict["labels"])): #tbins
            for j in range(len(data_dict["labels"][i])):
                labels = data_dict["labels"][i][j]
                if isinstance(labels, np.ndarray):
                    labels['x'] = labels['x'] / self.zoom_out_factor + self.zoom_out_x0
                    labels['y'] = labels['y'] / self.zoom_out_factor + self.zoom_out_y0
                    labels['w'] = labels['w'] / self.zoom_out_factor
                    labels['h'] = labels['h'] / self.zoom_out_factor
                    data_dict["labels"][i][j] = labels
        
        return data_dict
    
    def zoom_in(self, data_dict):
        
        zoom_window_h, zoom_window_w = int(self.height / self.zoom_in_factor), int(self.width / self.zoom_in_factor)
        
        zoom_canvas = data_dict['inputs'][..., self.zoom_in_y0:self.zoom_in_y0 + zoom_window_h, self.zoom_in_x0:self.zoom_in_x0 + zoom_window_w]
        
        t, n, c, h, w = zoom_canvas.size()
        zoom_canvas = zoom_canvas.view(n * t, c, h, w)
        output = interpolate(zoom_canvas, size=(self.height, self.width), mode='nearest-exact')
        output = output.view(t, n, c, self.height, self.width).contiguous()
        
        data_dict['inputs'] = output
        
        # label:
        for i in range(len(data_dict["labels"])): #tbins
            for j in range(len(data_dict["labels"][i])):
                labels = data_dict["labels"][i][j]
                # if len(labels) == 0:
                #     continue
                # else:
                if isinstance(labels, np.ndarray):
                    labels['x'] = (labels['x'] - self.zoom_in_x0) * self.zoom_in_factor
                    labels['y'] = (labels['y'] - self.zoom_in_y0) * self.zoom_in_factor
                    labels['w'] = labels['w'] * self.zoom_in_factor
                    labels['h'] = labels['h'] * self.zoom_in_factor
                    
                    # for label in labels:
                    #     if 0<= label['x'] + label['w']/2 <= self.width and 0<= label['y'] + label['h']/2 <= self.height:
                         
                    data_dict["labels"][i][j] = labels
        
        return data_dict