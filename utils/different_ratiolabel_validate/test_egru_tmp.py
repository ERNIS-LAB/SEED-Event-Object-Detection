import os,gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from functools import partial
import glob
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
import datetime
import numpy as np
np.set_printoptions(precision=16,threshold=np.inf)
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/shenqi/Master_thesis/Shenqi-MasterThseis/Lib')

# import metavision_ml_copy.data
from metavision_ml_copy.data import SequentialDataLoader
from metavision_ml_copy.data import box_processing as box_api
from metavision_ml_copy.detection.losses import DetectionLoss
from metavision_ml_copy.detection.feature_extractors import EGRUReLu_resnet_sparse, EMinRNN_ReLu, ELSTMReLu_resnet_sparse
from metavision_ml_copy.detection.single_stage_detector import SingleStageDetector
from metavision_ml_copy.metrics.coco_eval import CocoEvaluator

sys.path.append('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code')
# from metavision_ml.metrics.coco_eval import evaluate_detection
from utils.PropheseeToolbox.src.metrics.coco_eval import evaluate_detection
from utils.PropheseeToolbox.src.io.box_loading import reformat_boxes
from utils.PropheseeToolbox.src.io.box_filtering import filter_boxes
from metavision_sdk_core import EventBbox

torch.manual_seed(0)
np.random.seed(0)
scaler = torch.cuda.amp.GradScaler()


dataset_type = 'gen1'

if dataset_type == 'gen1':
    dataset_path = '/media/shenqi/data/Gen1_multi_timesurface_FromDat'
    # dataset_path = '/media/shenqi/data_0/Gen1_multi_timesurface_small'
elif dataset_type == 'gen4':
    dataset_path = '/media/shenqi/data/Gen4_multi_timesurface_FromDat'


label_map_path = os.path.join(dataset_path, 'label_map_dictionary.json')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_gt_file_paths = sorted(glob.glob(train_path+'/*.npy'))
train_data_file_paths = sorted(glob.glob(train_path+'/*.h5'))
test_gt_file_paths = sorted(glob.glob(test_path+'/*.npy'))
test_data_file_paths = sorted(glob.glob(test_path+'/*.h5'))
val_gt_file_paths = sorted(glob.glob(val_path+'/*.npy'))
val_data_file_paths = sorted(glob.glob(val_path+'/*.h5'))

print('train:',len(train_gt_file_paths), len(train_data_file_paths))
print('val:',len(val_gt_file_paths), len(val_data_file_paths))
print('test:',len(test_gt_file_paths), len(test_data_file_paths))

if dataset_type == 'gen1':
    wanted_keys = [ 'car', 'pedestrian']
    height, width = 240, 304
    min_box_diag_network = 30
elif dataset_type == 'gen4':
    wanted_keys = ['pedestrian', 'two wheeler', 'car']
    height, width = 360, 640
    min_box_diag_network = 60

class_lookup = box_api.create_class_lookup(label_map_path, wanted_keys)
print(class_lookup)

files = glob.glob(os.path.join(train_path, "*.h5"))
files_val = glob.glob(os.path.join(val_path, "*.h5"))
files_test = glob.glob(os.path.join(test_path, "*.h5"))

preprocess_function_name = "multi_channel_timesurface" 
delta_t = 50000
channels = 6  # histograms have two channels
num_tbins = 10
batch_size = 4
max_incr_per_pixel = 5
array_dim = [num_tbins, channels, height, width]


load_boxes_fn = partial(box_api.load_boxes, num_tbins=num_tbins, class_lookup=class_lookup, min_box_diag_network=min_box_diag_network)

seq_dataloader = SequentialDataLoader(files, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=8, padding=True, preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})

seq_dataloader_val = SequentialDataLoader(files_val, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=8, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
seq_dataloader_test = SequentialDataLoader(files_test, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=8, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
print(len(seq_dataloader))
print(len(seq_dataloader_val))
print(len(seq_dataloader_test))


model_name = 'EGRUReLu_resnet_sparse'  
# model_name =  'ELSTMReLu_resnet_sparse'   # 14,578,342 + 1,290,800 = 15.87M
# model_name = 'Vanilla_EventGRU_sparse_outC384'
detector = SingleStageDetector(feature_extractor=model_name, in_channels=channels, num_classes=len(wanted_keys), 
                                feature_base=16, feature_channels_out=64, anchor_list='PSEE_ANCHORS', dataset = dataset_type)

net = EGRUReLu_resnet_sparse(channels, base=16, cout=64, dataset = dataset_type, pruning = False)
box_coder = detector.box_coder
pd = detector.rpn


def bboxes_to_box_vectors(bbox):
    if isinstance(bbox, list):
        return [bboxes_to_box_vectors(item) for item in bbox]
    elif isinstance(bbox, np.ndarray) and bbox.dtype != np.float32:
        return box_api.bboxes_to_box_vectors(bbox)
    else:
        return bbox
    
class Trainer:
    def __init__(self, detector,model: nn.Module, BoxHead, seq_dataloader_train, seq_dataloader_val,seq_dataloader_test):
        self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        # print(self.model)
        self.detector = detector
        self.pd = BoxHead.to(self.device)

        
        self.seq_dataloader_train = seq_dataloader_train
        self.seq_dataloader_val = seq_dataloader_val
        self.seq_dataloader_test = seq_dataloader_test.cuda()
        self.criterion = DetectionLoss("softmax_focal_loss")
        
        self.cnt = 0
        self.cnt_train = 0
        self.cnt_val = 0
        self.label_map = ['background'] + wanted_keys

        if dataset_type == 'gen1':
            self.model.load_state_dict(torch.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/RED_based/models_4paper/Gen1_64EGRUReLu_resnet_sparse_4_20/34_model.pth',map_location=torch.device(self.device)))
            self.pd.load_state_dict(torch.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/RED_based/models_4paper/Gen1_64EGRUReLu_resnet_sparse_4_20/34_pd.pth',map_location=torch.device(self.device)))
        elif dataset_type == 'gen4':
            self.model.load_state_dict(torch.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/RED_based/models_4paper/NoInverse_EGRUReLu_resnet_sparse_4_20/28_model.pth',map_location=torch.device(self.device)))
            self.pd.load_state_dict(torch.load('/home/shenqi/Master_thesis/Shenqi-MasterThseis/code/RED_based/models_4paper/NoInverse_EGRUReLu_resnet_sparse_4_20/28_pd.pth',map_location=torch.device(self.device)))

    
    def val_epoch(self, seq_dataloader_val,epochs):
        
        self.model.eval()
        self.pd.eval()
        output_val_list = []
        loss_val_mean = 0
        self.cnt_val = 0
        mean_activity_egru_ave = [0] * self.model.levels
        mean_activity_conv1_ave = [0] * 7
        mean_activity_egruRelu_ave = [0] * 3

        
        # total_num_fire = [torch.zeros(num_tbins,batch_size,256,45,80).to('cuda'), torch.zeros(num_tbins,batch_size,256,23,40).to('cuda'), torch.zeros(num_tbins,batch_size,256,12,20).to('cuda')]
        
        with tqdm(total=len(seq_dataloader_val), desc=f'Validation',ncols=120) as pbar:
                        
            for data in seq_dataloader_val:
                sys.stdout.flush()
                with torch.no_grad():
                    self.cnt_val += 1
                    data['inputs'] = data['inputs'].to(device=self.device)
                                        
                                
                    output_val,mean_activity, mean_activity_conv1, output_gates_val, mean_activity_egru_relu = self.inference_step(data)
                    
                    # for i in range(len(total_num_fire)):
                    #     total_num_fire[i] += output_gates_val[i]
                    
                    output_val_list.append(output_val) 
                    
                    for i in range(self.model.levels):
                        mean_activity_egru_ave[i] += mean_activity[i].item()
                    for i in range(7):
                        mean_activity_conv1_ave[i] +=  mean_activity_conv1[i].item()
                    for i in range(self.model.levels):
                        mean_activity_egruRelu_ave[i] += mean_activity_egru_relu[i].item()
                    
                    
                    pbar.update(1)
                    
            mean_activity_egru_ave = [item/self.cnt_val for item in mean_activity_egru_ave]
            mean_activity_conv1_ave = [item/self.cnt_val for item in mean_activity_conv1_ave]
            mean_activity_egruRelu_ave = [item/self.cnt_val for item in mean_activity_egruRelu_ave]
            
            
        dt_detections = defaultdict(list)
        gt_detections = defaultdict(list)
        for item in output_val_list:
            for k, v in item['dt'].items():
                dt_detections[k].extend(v)
            for k, v in item['gt'].items():
                gt_detections[k].extend(v)
        
        if dataset_type == 'gen1':
            detction_output_path ='./dt_results/Gen1_egru/' + datetime.datetime.now().strftime("%m%d-%H%M") + '/'
        elif dataset_type == 'gen4':
            detction_output_path ='./dt_results/Gen4_egru/' + datetime.datetime.now().strftime("%m%d-%H%M") + '/'
            
        try:
            os.mkdir(detction_output_path)
        except FileExistsError:
                pass 
            
        for key in dt_detections:
            file_name = detction_output_path + key.split('/')[-1].split('.')[0] + '_dt_bbox.npy'
            np.save(file_name, np.concatenate(dt_detections[key])) 
        for key in gt_detections:
            file_name = detction_output_path + key.split('/')[-1].split('.')[0] + '_GroundTruth_bbox.npy'
            np.save(file_name, np.concatenate(gt_detections[key])) 
            
        test_gt_files = sorted(glob.glob(detction_output_path + '/*_GroundTruth_bbox.npy'))   
        gt_boxes_list = [np.load(p) for p in test_gt_files]
        test_dt_files = sorted(glob.glob(detction_output_path + '/*_dt_bbox.npy'))  
        result_boxes_list = [np.load(p) for p in test_dt_files]
        
        gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]
        result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
        
        if dataset_type == 'gen1':
            min_box_diag,min_box_side = 30, 10
        elif dataset_type == 'gen4':
            min_box_diag,min_box_side = 60, 20
        filter_boxes_fn = lambda x:filter_boxes(x, 500000, min_box_diag, min_box_side)
        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)
 
        evaluate_detection(gt_boxes_list, result_boxes_list,classes=self.label_map, height=height, width=width,time_tol=40000)
        
        
        
        
        
        # coco_val_result = self.inference_epoch_end(output_val_list)
        coco_val_result = None
        print(coco_val_result, '\n mean_activity_egru_ave:', mean_activity_egru_ave, '\n mean_activity_conv1', mean_activity_conv1_ave, '\n mean_activity_egruRelu', mean_activity_egruRelu_ave)
   
                
    
    def fit(self, epochs: int):
              
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            
            metrics_test = self.val_epoch(self.seq_dataloader_test,epoch)

            
                    

torch.cuda.empty_cache()
trainer = Trainer(detector, net, pd ,seq_dataloader, seq_dataloader_val,seq_dataloader_test)
trainer.fit(epochs=1)