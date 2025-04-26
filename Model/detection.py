import torch 
from collections import defaultdict
import numpy as np
import os
import glob
from metavision_ml.data import box_processing as box_api
from metavision_ml.metrics.coco_eval import CocoEvaluator
from metavision_sdk_core import EventBbox
from utils.PropheseeToolbox.src.metrics.coco_eval import evaluate_detection
from utils.PropheseeToolbox.src.io.box_loading import reformat_boxes
from utils.PropheseeToolbox.src.io.box_filtering import filter_boxes
import shutil  

def inference_step(batch, model, head, box_coder):
       
        # raise Exception
        with torch.no_grad():
            model.reset(batch["mask_keep_memory"])
           
            score_thresh=0.01
            
            nms_thresh=0.6
            max_boxes_per_input = 500

            features, events, neg_dis, conv1_outs, egruReluouts = model(batch['inputs'])
            
            output_gates_val = [torch.where(event == 0, torch.zeros_like(event), torch.ones_like(event)) for event in events]
            mean_activity = [torch.mean(output_gate_val).to('cuda') for output_gate_val in output_gates_val]
            output_gates_val_conv1 = [torch.where(conv1_out > 0, torch.ones_like(conv1_out), torch.zeros_like(conv1_out)) for conv1_out in conv1_outs]
            mean_activity_conv1 = [torch.mean(output_gate_val_conv1).to('cuda') for output_gate_val_conv1 in output_gates_val_conv1]
            output_egru_relus = [torch.where(egruReluout != 0, torch.ones_like(egruReluout), torch.zeros_like(egruReluout)) for egruReluout in egruReluouts]
            mean_activity_egru_relu = [torch.mean(output_egru_relu).to('cuda') for output_egru_relu in output_egru_relus]
            
            loc_preds, cls_preds, box_hids, cls_hids = head(features)
            output_gates_box = [torch.where(box_hid > 0, torch.ones_like(box_hid), torch.zeros_like(box_hid)) for box_hid in box_hids]
            box_hid_mean = [torch.mean(output_gate_box).to('cuda') for output_gate_box in output_gates_box]
            output_gates_cls = [torch.where(cls_hid > 0, torch.ones_like(cls_hid), torch.zeros_like(cls_hid)) for cls_hid in cls_hids]
            cls_hid_mean = [torch.mean(output_gate_cls).to('cuda') for output_gate_cls in output_gates_cls]
            
            scores = head.get_scores(cls_preds)
            
            scores = scores.to('cpu')
            for i, feat in enumerate(features):
                features[i] = features[i].to('cpu')
            inputs = batch['inputs'].to('cpu')
            loc_preds = loc_preds.to('cpu')

            preds = box_coder.decode(features, inputs, loc_preds, scores, batch_size=batch['inputs'].shape[1], score_thresh=score_thresh,
                                     nms_thresh=nms_thresh, max_boxes_per_input=max_boxes_per_input)
           
        dt_dic, gt_dic = accumulate_predictions(
            preds,
            batch["labels"],
            batch["video_infos"],
            batch["frame_is_labeled"])
        
        return {'dt': dt_dic, 'gt': gt_dic}, mean_activity, mean_activity_conv1, output_gates_val_conv1, mean_activity_egru_relu, box_hid_mean, cls_hid_mean

def accumulate_predictions(preds, targets, video_infos, frame_is_labeled):
       
        dt_detections = {}
        gt_detections = {}
        
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                gt_boxes = targets[t][i]
                # pred = preds[t][i]

                video_info, tbin_start, _ = video_infos[i]

                if video_info.padding or frame_is_labeled[t, i] == False:
                    continue

                name = video_info.path
            
                if name not in gt_detections:
                    gt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                assert video_info.start_ts == 0
                ts = tbin_start + t * video_info.delta_t

                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                if gt_boxes.dtype == np.float32:
                    gt_boxes = box_api.box_vectors_to_bboxes(gt_boxes[:, :4], gt_boxes[:, 4], ts=ts)
               
                if len(gt_boxes):
                    gt_boxes["t"] = ts
                    gt_detections[name].append(gt_boxes)
                else:
                    gt_detections[name].append(np.zeros((0), dtype=EventBbox))
                 
        for t in range(len(preds)):
            for i in range(len(preds[t])):
        
                video_info, tbin_start, _ = video_infos[i]
                if video_info.padding:
                    continue
                ts = tbin_start + t * 50000  
                
                name = video_info.path            
                if name not in dt_detections:
                    dt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                
                pred = preds[t][i]
                if pred['boxes'] is not None and len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().data.numpy()
                    labels = pred['labels'].cpu().data.numpy()
                    scores = pred['scores'].cpu().data.numpy()
                    dt_boxes = box_api.box_vectors_to_bboxes(boxes, labels, scores, ts=ts)
                    dt_detections[name].append(dt_boxes)
                else:
                    dt_detections[name].append(np.zeros((0), dtype=EventBbox))
     
        return dt_detections, gt_detections

def evaluate(output_val_list, dataloader):
    dt_detections = defaultdict(list)
    gt_detections = defaultdict(list)
    for item in output_val_list:
        for k, v in item['dt'].items():
            dt_detections[k].extend(v)
        for k, v in item['gt'].items():
            gt_detections[k].extend(v)
    
    if dataloader.dataset_type == 'gen1':
        detction_output_path ='./Log/dt_results/gen1/gru/'
    elif dataloader.dataset_type == 'gen4':
        detction_output_path ='./Log/dt_results/gen4/1000/'
        
    try:
        os.makedirs(detction_output_path)
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
   
    filter_boxes_fn = lambda x:filter_boxes(x, 500000, dataloader.min_box_diag_network, dataloader.min_box_diag_network/3)
    gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
    result_boxes_list = map(filter_boxes_fn, result_boxes_list)

    evaluate_detection(gt_boxes_list, result_boxes_list,classes=['background'] + dataloader.wanted_keys, height=dataloader.height, width=dataloader.width,time_tol=40000)
    shutil.rmtree(detction_output_path)  

