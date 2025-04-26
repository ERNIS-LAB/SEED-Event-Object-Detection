import sys,os

from tqdm import tqdm
import torch
import torch.nn as nn
from metavision_ml.detection.anchors import Anchors

from Model.feature_extractor import SEED_EventGRU
from Model.ssd_head import BoxHead
from Model.detection import inference_step
from Model.detection import evaluate
from utils.dataloader import seq_dataloader
import utils.data_augmentation as data_aug


##########################################################################################
dataset_path = '/media/shenqi/data/Gen4_multi_timesurface_FromDat'
dataset_type = 'gen4'
dataloader = seq_dataloader(dataset_path = dataset_path, dataset_type = dataset_type, num_tbins = 12, batch_size = 5, channels = 6)

cout = 256
threshold_group = 1
##########################################################################################
net = SEED_EventGRU(dataloader.channels, base=int(cout/16), cout=cout, dataset = dataset_type, pruning = False)
box_coder = Anchors(num_levels=net.levels, anchor_list='PSEE_ANCHORS', variances=[0.1, 0.2])
ssd_head = BoxHead(net.cout, box_coder.num_anchors, len(dataloader.wanted_keys)+1, n_layers=0)


net.load_state_dict(torch.load('./Saved_Model/gen4/SEED_Event_GRU/27_model.pth',map_location=torch.device('cuda')))
ssd_head.load_state_dict(torch.load('./Saved_Model/gen4/SEED_Event_GRU/27_model.pth',map_location=torch.device('cuda')))
# net.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/EMinGRU_ReLUFuseDownsampleConv_ConditionalConv_256_norelu_n15b8/48_model.pth',map_location=torch.device('cuda')))
# ssd_head.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/EMinGRU_ReLUFuseDownsampleConv_ConditionalConv_256_norelu_n15b8/48_pd.pth',map_location=torch.device('cuda')))

net.eval().to('cuda')
ssd_head.eval().to('cuda')

augment = data_aug.data_augmentation(dataset_type= dataset_type)

output_val_list = []
cnt_val = 0
mean_activity_egru_ave = [0] * net.levels
mean_activity_conv1_ave = [0] * 7
mean_activity_egruRelu_ave = [0] * net.levels
box_hid_mean_ave = [0] * net.levels
cls_hid_mean_ave = [0] * net.levels

with tqdm(total=len(dataloader.seq_dataloader_test), desc=f'Validation',ncols=120) as pbar:
            
    for data in dataloader.seq_dataloader_test:
        sys.stdout.flush()
        with torch.no_grad():
            cnt_val += 1
            data['inputs'] = data['inputs'].to(device='cuda')
            
            if data['frame_is_labeled'].sum().item() != 0:
                data = augment(data, only_vertical_move=True)  
                
            output_val,mean_activity, mean_activity_conv1, output_gates_val, mean_activity_egru_relu, box_hid_mean, cls_hid_mean = inference_step(data,net,ssd_head,box_coder)
            
            output_val_list.append(output_val) 
            
            for i in range(net.levels):
                mean_activity_egru_ave[i] += mean_activity[i].item()
            for i in range(7):
                mean_activity_conv1_ave[i] +=  mean_activity_conv1[i].item()
            for i in range(net.levels):
                mean_activity_egruRelu_ave[i] += mean_activity_egru_relu[i].item()
            for i in range(net.levels):
                box_hid_mean_ave[i] += box_hid_mean[i].item()
                cls_hid_mean_ave[i] += cls_hid_mean[i].item()
            
            pbar.update(1)
        
mean_activity_egru_ave = [item/cnt_val for item in mean_activity_egru_ave]
mean_activity_conv1_ave = [item/cnt_val for item in mean_activity_conv1_ave]
mean_activity_egruRelu_ave = [item/cnt_val for item in mean_activity_egruRelu_ave]
box_hid_mean_ave = [item/cnt_val for item in box_hid_mean_ave]
cls_hid_mean_ave = [item/cnt_val for item in cls_hid_mean_ave]
     
evaluate(output_val_list, dataloader)       
        
        
print('mean_activity_egru_ave:', mean_activity_egru_ave, '\n mean_activity_conv1', mean_activity_conv1_ave, '\n mean_activity_egruRelu', mean_activity_egruRelu_ave, '\n box_hid_mean_ave', box_hid_mean_ave, '\n cls_hid_mean_ave', cls_hid_mean_ave)
   
