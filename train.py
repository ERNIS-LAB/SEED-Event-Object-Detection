import os
import sys
from tqdm import tqdm
from itertools import chain
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection.losses import DetectionLoss
from metavision_ml.data import box_processing as box_api
from Model.feature_extractor import SEED_EventGRU
from Model.ssd_head import BoxHead
from Model.detection import inference_step
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


def bboxes_to_box_vectors(bbox):
    if isinstance(bbox, list):
        return [bboxes_to_box_vectors(item) for item in bbox]
    elif isinstance(bbox, np.ndarray) and bbox.dtype != np.float32:
        return box_api.bboxes_to_box_vectors(bbox)
    else:
        return bbox
    
class Trainer:
    def __init__(self, model: nn.Module, BoxHead, dataloader):
        self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.ssd_head = BoxHead.to(self.device)
        self.augment = data_aug.data_augmentation(dataset_type= dataset_type)
        self.seq_dataloader_train = dataloader.seq_dataloader_train.cuda()
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.ssd_head.parameters()}],
                                           lr=0.00025, weight_decay=0)
       
        self.criterion = DetectionLoss("softmax_focal_loss")
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,  gamma=0.95)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = 0.00025, epochs = 35, steps_per_epoch =1)
        self.is_sparse = False
        
        self.resnet_sparse_factor = 25 # lower: more sparse
       
        self.cnt = 0
        self.cnt_train = 0
        self.cnt_val = 0
        
        # self.model.load_state_dict(torch.load('./Saved_Model/gen4/SEED_EventGRUFuseDownsampleConv_DoubleEventConv/35_model.pth',map_location=torch.device('cuda')))
        # self.ssd_head.load_state_dict(torch.load('./Saved_Model/gen4/SEED_EventGRUFuseDownsampleConv_DoubleEventConv/35_pd.pth',map_location=torch.device('cuda')))

        
        log_dir = './Log/Logging/' + str(dataset_type) + '/' + str(type(self.model).__name__) + datetime.datetime.now().strftime("%m%d-%H%M")
        if str(type(self.model).__name__) == 'SEED_EventGRUFuseDownsampleConv_DoubleConditionalConv':
            log_dir += '_threshold_group_' + str(threshold_group)
        if self.is_sparse:
            log_dir += '_sparsefactor_' + str(self.resnet_sparse_factor)
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            pass
            
        self.logger = SummaryWriter(log_dir=log_dir)
        self.l1_criterion = nn.L1Loss()
        self.l2_criterion = nn.MSELoss()
        

    def select_valid_frames(self, xs, targets, frame_is_labeled):
        frame_is_labeled = frame_is_labeled.bool()
        mask = frame_is_labeled.view(-1)
        xs = [item[mask] for item in xs]
        targets = [
            [targets[r][c] for c in range(len(frame_is_labeled[0])) if frame_is_labeled[r][c]]
            for r in range(len(frame_is_labeled))]
        return xs, targets
 
    # def log_gradients(self, model, step):
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             self.logger.add_histogram(f"gradients/{name}", param.grad, step)
    
    def compute_loss(self, x, targets, frame_is_labeled):
        
        xs, events, neg_dis, conv1_outs, egruReluouts = self.model(x)
        
        output_gates_val = [torch.where(event == 0, torch.zeros_like(event), torch.ones_like(event)) for event in events]
        mean_activity = [torch.mean(output_gate_val).to(self.device) for output_gate_val in output_gates_val]
        
        output_gates_val_conv1 = [torch.where(conv1_out > 0, torch.ones_like(conv1_out), torch.zeros_like(conv1_out)) for conv1_out in conv1_outs]
        mean_activity_conv1 = [torch.mean(output_gate_val_conv1).to(self.device) for output_gate_val_conv1 in output_gates_val_conv1]
        
        output_egru_relus = [torch.where(egruReluout != 0, torch.ones_like(egruReluout), torch.zeros_like(egruReluout)) for egruReluout in egruReluouts]
        mean_activity_egru_relu = [torch.mean(output_egru_relu).to(self.device) for output_egru_relu in output_egru_relus]
        
        resnet_egruRelu_activity_loss = torch.zeros(1).to(self.device)
        for i in conv1_outs:
            resnet_egruRelu_activity_loss += self.l1_criterion(i, torch.zeros_like(i))
        for i in egruReluouts:
            resnet_egruRelu_activity_loss += self.l1_criterion(i, torch.zeros_like(i))
        resnet_egruRelu_activity_loss = resnet_egruRelu_activity_loss/self.resnet_sparse_factor
        
        
        neg_distance_loss = torch.zeros(1).to(self.device)
        for i in neg_dis:
            neg_distance_loss += self.l2_criterion(i, torch.zeros_like(i))
        
        if frame_is_labeled.sum().item() == 0:
            del xs
            return None, mean_activity, neg_distance_loss, mean_activity_conv1, resnet_egruRelu_activity_loss, mean_activity_egru_relu
        
        xs, targets = self.select_valid_frames(xs, targets, frame_is_labeled)
        
        loc_preds, cls_preds, box_hids, cls_hids = self.ssd_head(xs)
        targets = list(chain.from_iterable(targets))
        targets = box_coder.encode(xs, x, targets)
        assert targets['cls'].shape[1] == cls_preds.shape[1]
        loss_dict = self.criterion(loc_preds, targets['loc'], cls_preds, targets["cls"])
        
        del xs
        
        return loss_dict, mean_activity, neg_distance_loss, mean_activity_conv1, resnet_egruRelu_activity_loss, mean_activity_egru_relu
    


    def train_epoch(self, seq_dataloader_train, epoch):
        
        seq_dataloader_train.dataset.shuffle()

        self.model.train()
        self.ssd_head.train()
        
        sys.stdout.flush() 
        
        drop_epoch = 0
        aa = len(seq_dataloader_train)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        self.logger.add_scalar('learning rate', self.optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        with tqdm(total=aa, desc=f'Training',ncols=120) as pbar:
                        
            for data in seq_dataloader_train:
                sys.stdout.flush()
                self.cnt_train += 1
                
                self.optimizer.zero_grad()
                data['inputs'] = data['inputs'].to(device=self.device)
    
                if data['frame_is_labeled'].sum().item() != 0:
                    data = self.augment(data)
                
                self.model.reset(torch.zeros_like(data['mask_keep_memory']).to(device=self.device))  
                
                targets = bboxes_to_box_vectors(data['labels'])
                
                loss_dict, mean_activity, neg_distance_loss, mean_activity_conv1, resnet_egruRelu_activity_loss, mean_activity_egru_relu = self.compute_loss(data['inputs'], targets, data['frame_is_labeled'])
        
                if loss_dict is None:
                    continue
                
                loss = sum([value for key, value in loss_dict.items()])                
                if self.is_sparse:
                    loss += resnet_egruRelu_activity_loss.squeeze()
                
                
                if torch.isnan(neg_distance_loss.squeeze()):
                    print(data["video_infos"])
                    raise Exception
                
                if torch.isnan(resnet_egruRelu_activity_loss.squeeze()):
                    print(data["video_infos"])
                    raise Exception
                
                if torch.isnan(loss):
                    print(data["video_infos"])
                    raise Exception

                loss.backward()
                self.optimizer.step()
                
                step_metrics = {
                    'loss': loss.item(),
                    'loc_loss': loss_dict['loc_loss'].item(),
                    'cls_loss': loss_dict['cls_loss'].item()
                }

                pbar.set_postfix(**step_metrics)
                pbar.update(1)
                
                self.logger.add_scalar('resnet activity loss', resnet_egruRelu_activity_loss.item(),self.cnt_train)
                self.logger.add_scalar('neg_distance_loss loss', neg_distance_loss.item(),self.cnt_train)
                self.logger.add_scalar('train loss', loss.item(),self.cnt_train)
            
                self.logger.add_scalar('mean_activity_conv1_0', mean_activity_conv1[0].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_1_res', mean_activity_conv1[1].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_1', mean_activity_conv1[2].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_2_res', mean_activity_conv1[3].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_2', mean_activity_conv1[4].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_3_res', mean_activity_conv1[5].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_conv1_3', mean_activity_conv1[6].item(),self.cnt_train )
                
                self.logger.add_scalar('mean_activity_0', mean_activity[0].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_1', mean_activity[1].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_2', mean_activity[2].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_egru_relu_0', mean_activity_egru_relu[0].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_egru_relu_1', mean_activity_egru_relu[1].item(),self.cnt_train )
                self.logger.add_scalar('mean_activity_egru_relu_2', mean_activity_egru_relu[2].item(),self.cnt_train )
                # self.log_gradients(self.model, self.cnt_train)
               
                del loss, neg_distance_loss, resnet_egruRelu_activity_loss
                
    def save_checkpoint(self, epoch, model_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ssd_head_state_dict': self.ssd_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        checkpoint_path = os.path.join(model_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ssd_head.load_state_dict(checkpoint['ssd_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

        # Ensure the optimizer states are moved to the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")              
    
    def fit(self, epochs: int, resume_from=None):
       
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            self.start_epoch = 1
            
        for epoch in range(self.start_epoch, epochs+1):
            print(f'Epoch {epoch}')
            self.train_epoch(self.seq_dataloader_train, epoch)
            
            model_path = './Saved_Model/new_gen4/' + str(dataset_type) + '/' + str(type(self.model).__name__) 
            if str(type(self.model).__name__) == 'SEED_EventGRUFuseDownsampleConv_DoubleConditionalConv':
                model_path += '_threshold_group_' + str(threshold_group)
            if self.is_sparse:
                model_path += '_sparsefactor_' + str(self.resnet_sparse_factor)
            
            try:
                os.makedirs(model_path)
            except FileExistsError:
                pass
            
            path_model = model_path +'/' + str(epoch) + '_model.pth'
            path_ssd_head = model_path +'/' + str(epoch) + '_pd.pth'
            
            torch.save(self.model.state_dict(),path_model)
            torch.save(self.ssd_head.state_dict(),path_ssd_head)
            
            self.scheduler.step() 
            
            self.save_checkpoint(epoch, model_path)
                    

torch.cuda.empty_cache()
trainer = Trainer(net, ssd_head ,dataloader)
trainer.fit(epochs=35, resume_from = None)