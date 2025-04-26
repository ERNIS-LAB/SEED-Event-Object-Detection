import sys,os,time
sys.path.append('/home/shenqi/Master_thesis/SEED')
from tqdm import tqdm
from functools import partial
import numpy as np
import cv2
import torch
import torch.nn as nn
from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection_tracking.display_frame import draw_box_events
import matplotlib.pyplot as plt
from Model.feature_extractor import EMinGRU_ReLUFuseDownsampleConv_EventConv,ConditionalInput_ELSTM_ConditionalConv
from Model.ssd_head import BoxHead
from Model.detection import inference_step,evaluate
from utils.dataloader import seq_dataloader
import utils.data_augmentation as data_aug
from skvideo.io import FFmpegWriter

dataset_path = '/media/shenqi/data/Gen4_multi_timesurface_FromDat_super_small'
dataset_type = 'gen4'
dataloader = seq_dataloader(dataset_path = dataset_path, dataset_type = dataset_type, num_tbins = 1, batch_size = 1, channels = 6)

cout = 256
######################################################################################################################################
conditional_net = ConditionalInput_ELSTM_ConditionalConv(dataloader.channels, base=int(cout/16), cout=cout, dataset = dataset_type, pruning = False)
conditional_box_coder = Anchors(num_levels=conditional_net.levels, anchor_list='PSEE_ANCHORS', variances=[0.1, 0.2])
conditional_ssd_head = BoxHead(conditional_net.cout, conditional_box_coder.num_anchors, len(dataloader.wanted_keys)+1, n_layers=0)

conditional_net.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/ConditionalInput_ELSTM_ConditionalConv/49_model.pth',map_location=torch.device('cuda')))
conditional_ssd_head.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/ConditionalInput_ELSTM_ConditionalConv/49_pd.pth',map_location=torch.device('cuda')))

conditional_net.eval().to('cuda')
conditional_ssd_head.eval().to('cuda')
######################################################################################################################################
event_net = EMinGRU_ReLUFuseDownsampleConv_EventConv(dataloader.channels, base=int(cout/16), cout=cout, dataset = dataset_type, pruning = False)
event_box_coder = Anchors(num_levels=event_net.levels, anchor_list='PSEE_ANCHORS', variances=[0.1, 0.2])
event_ssd_head = BoxHead(event_net.cout, event_box_coder.num_anchors, len(dataloader.wanted_keys)+1, n_layers=0)

event_net.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/EMinGRU_ReLUFuseDownsampleConv_EventConv/44_model.pth',map_location=torch.device('cuda')))
event_ssd_head.load_state_dict(torch.load('/home/shenqi/Master_thesis/SEED/Saved_Model/new_gen4/EMinGRU_ReLUFuseDownsampleConv_EventConv/44_pd.pth',map_location=torch.device('cuda')))

event_net.eval().to('cuda')
event_ssd_head.eval().to('cuda')
######################################################################################################################################
augment = data_aug.data_augmentation(dataset_type= dataset_type)

viz_labels = partial(draw_box_events, label_map=['background']+dataloader.wanted_keys, thickness = 2)
# video_writer = FFmpegWriter('EMGU_condition.mp4', outputdict={'-vcodec': 'libx264', '-crf': '20', '-preset': 'veryslow','-r': '20'})
size_x = 3
size_y = 1
height_scaled = 360
width_scaled = 640
frame = np.zeros((size_y * height_scaled, width_scaled * size_x, 3), dtype=np.uint8)

event_conv1 = [None] * 7
event_conv2 = [None] * 3

with tqdm(total=len(dataloader.seq_dataloader_val)) as pbar:
    with torch.no_grad():     
        for ind, data in enumerate(dataloader.seq_dataloader_val):
            pbar.update(1)
            
            mask = data["mask_keep_memory"]
            metadata = dataloader.seq_dataloader_val.dataset.get_batch_metadata(ind)
            
            data['inputs'] = data['inputs'].to(device='cuda')
            # if data['frame_is_labeled'].sum().item() != 0:
            #     data = augment(data, only_vertical_move=True)  
            
            batch = data["inputs"]
            # data["inputs"] = torch.zeros_like(data["inputs"]).to(device='cuda')
            output_conditional_emgu, output_gates_val, mean_activity_conv1, output_gates_val_conv1, mean_activity_egru_relu, box_hid_mean, cls_hid_mean = inference_step(data,conditional_net,conditional_ssd_head,conditional_box_coder)
            output_event_emgu,*_ = inference_step(data,event_net,event_ssd_head,event_box_coder)
            
            value_output_conditional_dt_emguf = list(output_conditional_emgu['dt'].values())
            value_output_event_dt_emguf = list(output_event_emgu['dt'].values())
            
            for i in range(7):
                
                if event_conv1[i] == None:
                    event_conv1[i] = output_gates_val_conv1[i][0].sum(dim=0)
                else:
                    event_conv1[i] += output_gates_val_conv1[i][0].sum(dim=0)
            for i in range(3):
                if event_conv2[i] == None:
                    event_conv2[i] = output_gates_val[i][0][0].sum(dim=0)
                else:
                    event_conv2[i] += output_gates_val[i][0][0].sum(dim=0)
            
            index = 0
            t = 0   
            im = batch[t][index]
            im = im.cpu().numpy()
            
            
            y, x = divmod(index, size_x)
            img = dataloader.seq_dataloader_val.get_vis_func()(im)
            
            img_conditional_emgu = img.copy()
            img_event_emgu = img.copy()
        
            if viz_labels is not None:
                labels = data["labels"][t][index]
                img = viz_labels(img, labels)
                
                if(len(value_output_conditional_dt_emguf)<1):
                    img_conditional_emgu = viz_labels(img_conditional_emgu, [])
                else:
                    img_conditional_emgu = viz_labels(img_conditional_emgu, value_output_conditional_dt_emguf[0][1])
                
                if(len(value_output_event_dt_emguf)<1):
                    img_event_emgu = viz_labels(img_event_emgu, [])
                else:
                    img_event_emgu = viz_labels(img_event_emgu, value_output_event_dt_emguf[0][1])
                
                
            if t <= 1 and not mask[index]:
                # mark the beginning of a sequence with a red square
                img[:10, :10, 0] = 222
            name = metadata[index][0].path.split('/')[-1]
            cv2.putText(img, name, (int(0.05 * (width_scaled)), int(0.94 * (height_scaled))),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (50, 240, 12))
            

            frame[y * (height_scaled):(y + 1) * (height_scaled),
                    x * (width_scaled): (x + 1) * (width_scaled)] = img
            frame[y * (height_scaled):(y + 1) * (height_scaled),
                (x+1) * (width_scaled): (x + 2) * (width_scaled)] = img_conditional_emgu
            frame[y * (height_scaled):(y + 1) * (height_scaled),
                (x+2) * (width_scaled): (x + 3) * (width_scaled)] = img_event_emgu
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # time.sleep(0.1)
    #         video_writer.writeFrame(frame)
            
fig = plt.figure(figsize=(20, 10))  

for i in range(7):
    ax = fig.add_subplot(2, 7, i+1)  
    im = ax.imshow(event_conv1[i].cpu().numpy(), cmap='viridis')
    ax.set_title(f'conv1_{i}')
    plt.colorbar(im, ax=ax)
   


for i in range(3):
    ax = fig.add_subplot(2, 7, i+8)  
    im = ax.imshow(event_conv2[i].cpu().numpy(), cmap='viridis')
    ax.set_title(f'conv2_{i}')
    plt.colorbar(im, ax=ax)
    print(i)
    print(event_conv2[i].cpu().numpy())

plt.tight_layout()  
plt.show()
