#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet-B0 多任務循序學習完整解決方案
作者: 程式夥伴 (基於使用者需求與參考資料)
日期: 2025/06/05 (修改為單次 Pass 流程，整合特徵蒸餾與精細化差分學習率)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF_func
from torchvision.ops import nms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt # 雖然未使用，但保留
import warnings
import math
import random

# 從 torchmetrics 導入 (假設已安裝)
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from torchmetrics import JaccardIndex, Accuracy as TorchmetricsAccuracy 
except ImportError:
    print("警告: torchmetrics 未安裝，mAP 和 mIoU 將無法準確計算。請執行: pip install torchmetrics")
    class MeanAveragePrecision:
        def __init__(self, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def compute(self): return {'map': torch.tensor(0.0), 'map_50': torch.tensor(0.0)}
        def reset(self): pass
    class JaccardIndex:
        def __init__(self, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def compute(self): return torch.tensor(0.0)
        def reset(self): pass
    class TorchmetricsAccuracy: 
        def __init__(self, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def compute(self): return torch.tensor(0.0)
        def reset(self): pass

warnings.filterwarnings('ignore')

from models import UnifiedMultiTaskModel # 確保 models.py 是最新版本 (包含返回中間特徵和可選的增強偵測頭)

# --- 全域常數 ---
NUM_DET_CLASSES = 10
NUM_SEG_CLASSES = 20
NUM_CLS_CLASSES = 10
IGNORE_INDEX_SEG = -100 
TARGET_IMG_SIZE = 224
GRID_SIZE_S = TARGET_IMG_SIZE // 16
BOXES_PER_CELL_B = 1


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # logger 可能尚未初始化，所以這裡用 print
    print(f"隨機種子已設定為: {seed_value}")

# --- Dataset 類別定義 (與你提供的一致) ---
# ... (CocoDetectionDataset, VOCSegmentationDataset 的代碼不變，請保留你已有的)
class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, common_transforms=None, specific_transforms=None):
        self.image_dir = image_dir
        self.common_transforms = common_transforms
        self.specific_transforms = specific_transforms
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images_log = self.coco_data['images']
        self.img_to_anns = {}
        if 'annotations' in self.coco_data:
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns: self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.images_log}

    def __getitem__(self, idx):
        img_log = self.images_log[idx]
        img_id = img_log['id']
        img_filename = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.image_dir, img_filename)
        try: image = Image.open(img_path).convert("RGB")
        except FileNotFoundError: return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), {'boxes': torch.zeros((0,4), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.long)}

        target = {'boxes': [], 'labels': []}
        original_w, original_h = image.size
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                bbox = ann['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                xmin, ymin, xmax, ymax = max(0, float(xmin)), max(0, float(ymin)), min(original_w, float(xmax)), min(original_h, float(ymax))
                if xmax > xmin and ymax > ymin:
                    target['boxes'].append([xmin, ymin, xmax, ymax])
                    target['labels'].append(ann['category_id'] - 1) 
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        if self.common_transforms:
            image, target_transformed = self.common_transforms(image, {'boxes': target['boxes'].clone(), 'labels': target['labels'].clone()})
            target['boxes'] = target_transformed['boxes']
            target['labels'] = target_transformed['labels']
        
        if self.specific_transforms: 
            image = self.specific_transforms(image)

        if not isinstance(target['boxes'], torch.Tensor): target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
        if not isinstance(target['labels'], torch.Tensor): target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        
        return image, target

    def __len__(self): return len(self.images_log)

class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set_file, common_transforms=None, specific_transforms=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass')
        self.common_transforms = common_transforms
        self.specific_transforms = specific_transforms
        image_set_path = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', image_set_file)
        try:
            with open(image_set_path, 'r') as f: self.image_filenames = [line.strip() for line in f.readlines()]
        except FileNotFoundError: self.image_filenames = []

    def __getitem__(self, idx):
        if not self.image_filenames: return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), torch.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE), dtype=torch.long)
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except FileNotFoundError: return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), torch.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE), dtype=torch.long)
        
        if self.common_transforms:
            image, mask = self.common_transforms(image, mask)
        
        if self.specific_transforms:
            image = self.specific_transforms(image)
        
        if isinstance(mask, Image.Image):
            mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        else: 
            mask_tensor = mask 
            
        return image, mask_tensor

    def __len__(self): return len(self.image_filenames)


# --- 簡易日誌記錄器 ---
class SimpleLogger:
    def __init__(self, log_file="training_log_single_pass.txt"): # 修改日誌檔名
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log Started (Single Pass): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write("="*30 + "\n")
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f: f.write(message + "\n")

logger = SimpleLogger()

# --- 設定隨機種子 ---
SEED = 42 
set_seed(SEED)
logger.log(f"隨機種子已通過 set_seed 函數設定為: {SEED}")


# --- 輔助函式: 同步轉換 ---
# ... (SynchronizedTransform 的代碼不變，請保留你已有的)
class SynchronizedTransform:
    def __init__(self, size, is_train):
        self.size = (size, size) if isinstance(size, int) else size
        self.is_train = is_train
    def __call__(self, image_pil, target=None): 
        original_w, original_h = image_pil.size
        image_pil_res = TF_func.resize(image_pil, self.size, interpolation=TF_func.InterpolationMode.BILINEAR)
        target_res = target 
        if isinstance(target, dict) and 'boxes' in target: 
            boxes = target['boxes'] 
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.size[0] / original_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.size[1] / original_h)
                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=self.size[0] - 1)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=self.size[1] - 1)
            target['boxes'] = boxes; target_res = target 
        elif isinstance(target, Image.Image): 
            target_res = TF_func.resize(target, self.size, interpolation=TF_func.InterpolationMode.NEAREST)
        if self.is_train and torch.rand(1) < 0.5:
            image_pil_res = TF_func.hflip(image_pil_res)
            if isinstance(target_res, dict) and 'boxes' in target_res: 
                boxes = target_res['boxes']
                if boxes.numel() > 0:
                    orig_x1 = boxes[:, 0].clone()
                    boxes[:, 0] = self.size[0] - boxes[:, 2]
                    boxes[:, 2] = self.size[0] - orig_x1
                target_res['boxes'] = boxes
            elif isinstance(target_res, Image.Image): target_res = TF_func.hflip(target_res)
        return image_pil_res, target_res

# --- NMS 和目標編碼 ---
# ... (yolo_v1_NMS, encode_yolo_targets 的代碼不變，請保留你已有的)
def yolo_v1_NMS(outputs, conf_thres=0.01, iou_thres=0.5, S=GRID_SIZE_S, B=BOXES_PER_CELL_B, C=NUM_DET_CLASSES, img_w=TARGET_IMG_SIZE, img_h=TARGET_IMG_SIZE):
    BATCH = outputs.shape[0]
    results = []
    for b_idx in range(BATCH):
        preds_single_image = outputs[b_idx] 
        boxes, scores, labels = [], [], []
        for row in range(S):
            for col in range(S):
                tx=torch.sigmoid(preds_single_image[0,row,col]); ty=torch.sigmoid(preds_single_image[1,row,col])
                pw_sqrt=preds_single_image[2,row,col]; ph_sqrt=preds_single_image[3,row,col]
                pconf=torch.sigmoid(preds_single_image[4,row,col])
                if pconf<conf_thres: continue
                cls_logits=preds_single_image[5:,row,col]; cls_probs=torch.softmax(cls_logits,dim=0)
                cls_score,cls_id=torch.max(cls_probs,dim=0); score=pconf*cls_score
                if score<conf_thres: continue
                cx=(col+tx.item())/S*img_w; cy=(row+ty.item())/S*img_h
                w=(pw_sqrt.item()**2)*img_w; h=(ph_sqrt.item()**2)*img_h
                x1=max(0,cx-w/2); y1=max(0,cy-h/2); x2=min(img_w-1,cx+w/2); y2=min(img_h-1,cy+h/2)
                if x2<=x1 or y2<=y1: continue
                boxes.append([x1,y1,x2,y2]); scores.append(score.item()); labels.append(cls_id.item())
        if not boxes: results.append({'boxes':torch.empty((0,4)),'labels':torch.empty((0,),dtype=torch.long),'scores':torch.empty((0,))}); continue
        boxes_t,scores_t,labels_t=torch.tensor(boxes,dtype=torch.float32),torch.tensor(scores,dtype=torch.float32),torch.tensor(labels,dtype=torch.long)
        keep=nms(boxes_t,scores_t,iou_thres)
        results.append({'boxes':boxes_t[keep],'labels':labels_t[keep],'scores':scores_t[keep]})
    return results

def encode_yolo_targets(target_list, S=GRID_SIZE_S, B=BOXES_PER_CELL_B, C=NUM_DET_CLASSES, img_size=(TARGET_IMG_SIZE,TARGET_IMG_SIZE), device='cpu'):
    batch_size = len(target_list)
    W_img, H_img = img_size
    yolo_target_tensor = torch.zeros(batch_size, B*5 + C, S, S, device=device)
    for b_idx, targets_dict in enumerate(target_list):
        boxes_xyxy = targets_dict['boxes'].cpu().numpy() 
        labels = targets_dict['labels'].cpu().numpy()
        for box_xyxy, label_idx in zip(boxes_xyxy, labels):
            xmin,ymin,xmax,ymax=box_xyxy; box_w_pixel,box_h_pixel=xmax-xmin,ymax-ymin
            cx_pixel,cy_pixel=xmin+box_w_pixel/2,ymin+box_h_pixel/2
            if box_w_pixel<=0 or box_h_pixel<=0: continue
            cx_norm,cy_norm=cx_pixel/W_img,cy_pixel/H_img; w_norm,h_norm=box_w_pixel/W_img,box_h_pixel/H_img
            col_idx,row_idx=min(S-1,int(cx_norm*S)),min(S-1,int(cy_norm*S))
            tx_target,ty_target=cx_norm*S-col_idx,cy_norm*S-row_idx
            tw_target,th_target=math.sqrt(max(w_norm,1e-8)),math.sqrt(max(h_norm,1e-8))
            obj_conf_channel_idx=4 
            if yolo_target_tensor[b_idx,obj_conf_channel_idx,row_idx,col_idx]==0:
                yolo_target_tensor[b_idx,0,row_idx,col_idx]=tx_target; yolo_target_tensor[b_idx,1,row_idx,col_idx]=ty_target
                yolo_target_tensor[b_idx,2,row_idx,col_idx]=tw_target; yolo_target_tensor[b_idx,3,row_idx,col_idx]=th_target
                yolo_target_tensor[b_idx,obj_conf_channel_idx,row_idx,col_idx]=1.0
                class_channel_offset=B*5; class_idx_int=int(label_idx)
                if 0<=class_idx_int<C: yolo_target_tensor[b_idx,class_channel_offset+class_idx_int,row_idx,col_idx]=1.0
    return yolo_target_tensor

# --- 損失函數 ---
# ... (DETLoss, SEGLoss, CLSLoss 的代碼不變，請保留你已有的)
class DETLoss(nn.Module):
    def __init__(self, S=GRID_SIZE_S, B=BOXES_PER_CELL_B, C=NUM_DET_CLASSES, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__(); self.S,self.B,self.C=S,B,C; self.lambda_coord,self.lambda_noobj=lambda_coord,lambda_noobj
        self.mse=nn.MSELoss(reduction='sum'); self.bce_logits=nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, preds, targets_encoded): 
        pred_tx=torch.sigmoid(preds[:,0,...]); pred_ty=torch.sigmoid(preds[:,1,...])
        pred_tw_sqrt=preds[:,2,...]; pred_th_sqrt=preds[:,3,...]
        pred_conf_logit=preds[:,4,...]; pred_cls_logits=preds[:,5:,...]
        target_tx=targets_encoded[:,0,...]; target_ty=targets_encoded[:,1,...]
        target_tw_sqrt=targets_encoded[:,2,...]; target_th_sqrt=targets_encoded[:,3,...]
        target_conf=targets_encoded[:,4,...]; target_cls_one_hot=targets_encoded[:,5:,...]
        obj_mask=target_conf>0; noobj_mask=target_conf==0
        num_obj=obj_mask.float().sum(); batch_size=preds.size(0)
        loss_tx=self.mse(pred_tx[obj_mask],target_tx[obj_mask]); loss_ty=self.mse(pred_ty[obj_mask],target_ty[obj_mask])
        loss_tw=self.mse(pred_tw_sqrt[obj_mask],target_tw_sqrt[obj_mask]); loss_th=self.mse(pred_th_sqrt[obj_mask],target_th_sqrt[obj_mask])
        coord_loss=self.lambda_coord*(loss_tx+loss_ty+loss_tw+loss_th)
        if num_obj>0: coord_loss=coord_loss/num_obj
        else: coord_loss=torch.tensor(0.0,device=preds.device,dtype=preds.dtype)
        obj_conf_loss=self.bce_logits(pred_conf_logit[obj_mask],target_conf[obj_mask])
        noobj_conf_loss=self.bce_logits(pred_conf_logit[noobj_mask],target_conf[noobj_mask])
        if num_obj>0: obj_conf_loss=obj_conf_loss/num_obj
        else: obj_conf_loss=torch.tensor(0.0,device=preds.device,dtype=preds.dtype)
        num_noobj_elements=noobj_mask.float().sum()
        if num_noobj_elements>0: noobj_conf_loss=noobj_conf_loss/num_noobj_elements
        else: noobj_conf_loss=torch.tensor(0.0,device=preds.device,dtype=preds.dtype)
        conf_loss=obj_conf_loss+self.lambda_noobj*noobj_conf_loss
        cls_loss_sum=self.mse(pred_cls_logits.permute(0,2,3,1)[obj_mask],target_cls_one_hot.permute(0,2,3,1)[obj_mask])
        if num_obj>0: cls_loss=cls_loss_sum/num_obj
        else: cls_loss=torch.tensor(0.0,device=preds.device,dtype=preds.dtype)
        total_loss=(coord_loss+conf_loss+cls_loss)/batch_size 
        return total_loss

class SEGLoss(nn.Module):
    def __init__(self,ignore_index=IGNORE_INDEX_SEG):super(SEGLoss,self).__init__();self.criterion=nn.CrossEntropyLoss(ignore_index=ignore_index)
    def remap_seg_targets(self,targets):new=targets.clone().long();new[(targets>0)&(targets<=20)]-=1;new[targets==0]=self.criterion.ignore_index;new[targets==255]=self.criterion.ignore_index;return new
    def forward(self,preds,targets):remapped_targets=self.remap_seg_targets(targets.to(preds.device));return self.criterion(preds,remapped_targets)
class CLSLoss(nn.Module):
    def __init__(self):super(CLSLoss,self).__init__();self.criterion=nn.CrossEntropyLoss()
    def forward(self,preds,targets):return self.criterion(preds,targets.long().to(preds.device))

# --- 評估器 ---
# ... (Evaluator, remap_seg_targets_for_metric 的代碼不變，請保留你已有的)
def remap_seg_targets_for_metric(targets,ignore_index_metric=IGNORE_INDEX_SEG,num_actual_classes=NUM_SEG_CLASSES):
    new=targets.clone().long();mask_fg=(new>=1)&(new<=20);new[mask_fg]=new[mask_fg]-1;new[targets==0]=ignore_index_metric;new[targets==255]=ignore_index_metric;return new
class Evaluator:
    def __init__(self,device,num_seg_classes=NUM_SEG_CLASSES,num_cls_classes=NUM_CLS_CLASSES,num_det_classes=NUM_DET_CLASSES,seg_ignore_index=IGNORE_INDEX_SEG):
        self.device=device;self.num_seg_classes=num_seg_classes;self.det_metric=MeanAveragePrecision(box_format="xywh",iou_type="bbox",class_metrics=False).to(device)
        self.seg_metric=JaccardIndex(task="multiclass",num_classes=self.num_seg_classes,ignore_index=seg_ignore_index,average='macro').to(device)
        self.cls_metric=TorchmetricsAccuracy(task="multiclass",num_classes=num_cls_classes,top_k=1).to(device);self.reset()
    def reset(self):self.det_metric.reset();self.seg_metric.reset();self.cls_metric.reset();self.results={}
    def xyxy_to_xywh_list(self,box_list_xyxy):
        new_list=[]
        for item_dict in box_list_xyxy:
            boxes_xyxy_tensor=item_dict['boxes'].cpu()
            if boxes_xyxy_tensor.numel()>0:xmin,ymin,xmax,ymax=boxes_xyxy_tensor.unbind(1);boxes_xywh_tensor=torch.stack([xmin,ymin,xmax-xmin,ymax-ymin],dim=1)
            else:boxes_xywh_tensor=torch.empty_like(boxes_xyxy_tensor)
            new_item_dict={'boxes':boxes_xywh_tensor,'labels':item_dict['labels'].cpu()}
            if 'scores' in item_dict:new_item_dict['scores']=item_dict['scores'].cpu()
            new_list.append(new_item_dict)
        return new_list
    def update(self,preds_model_output,targets,task:str):
        if task=='det':decoded_preds_nms=yolo_v1_NMS(preds_model_output.cpu());preds_xywh=self.xyxy_to_xywh_list(decoded_preds_nms);targets_xywh=self.xyxy_to_xywh_list(targets);self.det_metric.update(preds_xywh,targets_xywh)
        elif task=='seg':preds_logits=preds_model_output;pred_labels=torch.argmax(preds_logits,dim=1);remapped_target_labels=remap_seg_targets_for_metric(targets,ignore_index_metric=self.seg_metric.ignore_index,num_actual_classes=self.num_seg_classes);self.seg_metric.update(pred_labels.to(self.device),remapped_target_labels.to(self.device))
        elif task=='cls':cls_logits=preds_model_output;self.cls_metric.update(cls_logits.to(self.device),targets.to(self.device))
    def compute(self):
        results={};det_res=self.det_metric.compute();results['mAP']=det_res.get('map',torch.tensor(0.0)).item();results['mAP_50']=det_res.get('map_50',torch.tensor(0.0)).item()
        results['mIoU']=self.seg_metric.compute().item();results['Top1']=self.cls_metric.compute().item();self.results=results;return self.results

# --- 訓練與驗證迴圈 ---
# ... (train_one_epoch_seg, valid_one_epoch_seg, valid_one_epoch_det, valid_one_epoch_cls 的代碼不變)
def train_one_epoch_seg(model,dataloader,criterion,optimizer,device,logger,epoch,scaler,current_stage_name="Stage"):
    model.train();total_loss=0.0;progress_bar=tqdm(dataloader,desc=f"{current_stage_name} (Seg) Epoch {epoch+1} Train",leave=False)
    for i,(images,masks) in enumerate(progress_bar):
        images,masks=images.to(device),masks.to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',enabled=scaler.is_enabled()):outputs=model(images);seg_logits=outputs['seg'];loss=criterion(seg_logits,masks)
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();total_loss+=loss.item();progress_bar.set_postfix({'loss':f"{loss.item():.4f}"})
    avg_loss=total_loss/len(dataloader) if len(dataloader)>0 else 0;logger.log(f"{current_stage_name} (Seg) | Epoch: {epoch+1} | Average Training Loss: {avg_loss:.4f}");return avg_loss

def valid_one_epoch_seg(model,dataloader,criterion,evaluator,device,logger,epoch,stage_name="Validation (Seg)"):
    model.eval();total_loss=0.0;evaluator.reset();progress_bar=tqdm(dataloader,desc=f"{stage_name} Epoch {epoch+1} Val",leave=False)
    with torch.no_grad():
        for images,masks in progress_bar:
            images,masks=images.to(device),masks.to(device)
            with torch.autocast(device_type='cuda'):outputs=model(images);seg_logits=outputs['seg']
            loss=criterion(seg_logits,masks);total_loss+=loss.item();evaluator.update(seg_logits.cpu(),masks.cpu(),task='seg')
    avg_loss=total_loss/len(dataloader) if len(dataloader)>0 else 0;metrics=evaluator.compute();mIoU_val=metrics.get('mIoU',0.0);logger.log(f"{stage_name} | Epoch: {epoch+1} | Validation Loss: {avg_loss:.4f} | mIoU: {mIoU_val:.4f}");return avg_loss,mIoU_val

def valid_one_epoch_det(model,dataloader,evaluator,device,logger,epoch,stage_name="Validation (Det)"):
    model.eval();evaluator.reset();progress_bar=tqdm(dataloader,desc=f"{stage_name} Epoch {epoch+1} Val",leave=False)
    with torch.no_grad():
        for det_images_tuple,det_targets_list in progress_bar:
            det_images=torch.stack(det_images_tuple,dim=0).to(device)
            with torch.autocast(device_type='cuda'):outputs=model(det_images);det_preds_raw=outputs['det']
            evaluator.update(det_preds_raw.cpu(),det_targets_list,task='det')
    metrics=evaluator.compute();mAP_val=metrics.get('mAP_50',0.0);logger.log(f"{stage_name} | Epoch: {epoch+1} | Validation mAP@0.5: {mAP_val:.4f}");return mAP_val

def valid_one_epoch_cls(model,dataloader,evaluator,device,logger,epoch,stage_name="Validation (Cls)"):
    model.eval();evaluator.reset();progress_bar=tqdm(dataloader,desc=f"{stage_name} Epoch {epoch+1} Val",leave=False)
    with torch.no_grad():
        for images,labels in progress_bar:
            images,labels=images.to(device),labels.to(device)
            with torch.autocast(device_type='cuda'):outputs=model(images);cls_logits=outputs['cls']
            evaluator.update(cls_logits.cpu(),labels.cpu(),task='cls')
    metrics=evaluator.compute();top1_acc=metrics.get('Top1',0.0);logger.log(f"{stage_name} | Epoch: {epoch+1} | Validation Top-1 Acc: {top1_acc:.4f}");return top1_acc

# --- Epoch 訓練函數 (整合特徵蒸餾) ---
# train_one_epoch_det_distill_seg_cls (偵測為主，蒸餾分割和分類)
def train_one_epoch_det_distill_seg_cls(
    model, teacher_seg_model, teacher_cls_model, 
    det_dataloader, replay_seg_dataloader, replay_cls_loader, 
    criterion_det, criterion_seg_distill, criterion_cls_distill, 
    optimizer, device, logger, epoch, 
    distill_weight_seg, distill_weight_cls, 
    replay_freq, T, scaler, current_stage_name="Stage"
):
    model.train()
    if teacher_seg_model: teacher_seg_model.eval()
    if teacher_cls_model: teacher_cls_model.eval()
    total_loss_main, total_loss_distill_s, total_loss_distill_c = 0.0, 0.0, 0.0
    if replay_seg_dataloader: replay_seg_iter = iter(replay_seg_dataloader)
    else: replay_seg_iter = None
    if replay_cls_loader: replay_cls_iter = iter(replay_cls_loader)
    else: replay_cls_iter = None

    progress_bar = tqdm(det_dataloader, desc=f"{current_stage_name} (Det Main) Epoch {epoch+1} Train", leave=False)
    for i, (det_images_tuple, det_targets_list) in enumerate(progress_bar): 
        det_images = torch.stack(det_images_tuple, dim=0).to(device) 
        encoded_det_targets = encode_yolo_targets(det_targets_list, S=GRID_SIZE_S, B=BOXES_PER_CELL_B, C=NUM_DET_CLASSES, img_size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',enabled=scaler.is_enabled()):
            # 偵測是主要任務，學生模型在偵測數據上不需要返回中間特徵給自己蒸餾
            outputs_main = model(det_images, return_intermediate_feature_for_det=False)
            det_preds = outputs_main['det']
            loss_main = criterion_det(det_preds, encoded_det_targets)

            loss_distill_s_val = torch.tensor(0.0, device=device)
            if distill_weight_seg > 0 and teacher_seg_model and replay_seg_iter and (i % replay_freq == 0):
                try: replay_s_images, _ = next(replay_seg_iter)
                except StopIteration: replay_seg_iter = iter(replay_seg_dataloader); replay_s_images, _ = next(replay_seg_iter)
                replay_s_images = replay_s_images.to(device)
                with torch.no_grad(): teacher_s_logits = teacher_seg_model(replay_s_images)['seg']
                # 學生模型在分割 replay 數據上
                student_s_logits = model(replay_s_images, return_intermediate_feature_for_det=False)['seg'] 
                loss_distill_s_val = criterion_seg_distill(F.log_softmax(student_s_logits / T, dim=1), F.softmax(teacher_s_logits / T, dim=1)) * (T * T)
            
            loss_distill_c_val = torch.tensor(0.0, device=device)
            if distill_weight_cls > 0 and teacher_cls_model and replay_cls_iter and (i % replay_freq == 0):
                try: replay_c_images, _ = next(replay_cls_iter)
                except StopIteration: replay_cls_iter = iter(replay_cls_loader); replay_c_images, _ = next(replay_cls_iter)
                replay_c_images = replay_c_images.to(device)
                with torch.no_grad(): teacher_c_logits = teacher_cls_model(replay_c_images)['cls']
                # 學生模型在分類 replay 數據上
                student_c_logits = model(replay_c_images, return_intermediate_feature_for_det=False)['cls']
                loss_distill_c_val = criterion_cls_distill(F.log_softmax(student_c_logits / T, dim=1), F.softmax(teacher_c_logits / T, dim=1)) * (T*T)
            total_batch_loss = loss_main + distill_weight_seg * loss_distill_s_val + distill_weight_cls * loss_distill_c_val
        
        scaler.scale(total_batch_loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss_main += loss_main.item(); total_loss_distill_s += loss_distill_s_val.item() if isinstance(loss_distill_s_val, torch.Tensor) else loss_distill_s_val; total_loss_distill_c += loss_distill_c_val.item() if isinstance(loss_distill_c_val, torch.Tensor) else loss_distill_c_val
        progress_bar.set_postfix({'det_L': f"{loss_main.item():.4f}", 'seg_D': f"{loss_distill_s_val.item():.4f}", 'cls_D': f"{loss_distill_c_val.item():.4f}"})
    avg_loss_main = total_loss_main / len(det_dataloader) if len(det_dataloader) > 0 else 0
    num_distill_batches_s = (len(det_dataloader) // replay_freq if replay_freq > 0 and replay_seg_dataloader and len(det_dataloader) > 0 else 1); avg_loss_distill_s = total_loss_distill_s / max(1, num_distill_batches_s)
    num_distill_batches_c = (len(det_dataloader) // replay_freq if replay_freq > 0 and replay_cls_loader and len(det_dataloader) > 0 else 1); avg_loss_distill_c = total_loss_distill_c / max(1, num_distill_batches_c)
    logger.log(f"{current_stage_name} (Det Main) | Epoch: {epoch+1} | Avg Det Loss: {avg_loss_main:.4f} | Avg Seg Distill: {avg_loss_distill_s:.4f} | Avg Cls Distill: {avg_loss_distill_c:.4f}"); return avg_loss_main

# train_one_epoch_seg_distill_det_cls (分割為主，蒸餾偵測logits+feature，蒸餾分類)
def train_one_epoch_seg_distill_det_cls(
    model, teacher_det_model, teacher_cls_model,
    seg_dataloader, replay_det_loader, replay_cls_loader,
    criterion_seg, criterion_det_logits_distill, criterion_det_feature_distill, criterion_cls_distill,
    optimizer, device, logger, epoch,
    distill_weight_det_logits, distill_weight_det_feature, distill_weight_cls, 
    replay_freq, T, scaler, current_stage_name="Stage"
):
    model.train()
    if teacher_det_model: teacher_det_model.eval()
    if teacher_cls_model: teacher_cls_model.eval()
    total_loss_main, total_loss_distill_d_logits, total_loss_distill_d_feature, total_loss_distill_c = 0.0, 0.0, 0.0, 0.0
    if replay_det_loader: replay_det_iter = iter(replay_det_loader)
    else: replay_det_iter = None
    if replay_cls_loader: replay_cls_iter = iter(replay_cls_loader)
    else: replay_cls_iter = None

    progress_bar = tqdm(seg_dataloader, desc=f"{current_stage_name} (Seg Main) Epoch {epoch+1} Train", leave=False)
    for i, (main_images, main_masks) in enumerate(progress_bar):
        main_images, main_masks = main_images.to(device), main_masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            outputs = model(main_images, return_intermediate_feature_for_det=False) 
            seg_logits = outputs['seg']
            loss_main = criterion_seg(seg_logits, main_masks)
            
            loss_distill_d_logits_val, loss_distill_d_feature_val = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if (distill_weight_det_logits > 0 or distill_weight_det_feature > 0) and teacher_det_model and replay_det_iter and (i % replay_freq == 0):
                try: replay_d_images_tuple, _ = next(replay_det_iter)
                except StopIteration: replay_det_iter = iter(replay_det_loader); replay_d_images_tuple, _ = next(replay_det_iter)
                replay_d_images = torch.stack(replay_d_images_tuple, dim=0).to(device)
                student_outputs_full_det = model(replay_d_images, return_intermediate_feature_for_det=True)
                student_d_outputs_logits = student_outputs_full_det['det']
                student_d_inter_features = student_outputs_full_det.get('det_intermediate_feature')
                with torch.no_grad():
                    teacher_outputs_full_det = teacher_det_model(replay_d_images, return_intermediate_feature_for_det=True)
                    teacher_d_outputs_logits = teacher_outputs_full_det['det']
                    teacher_d_inter_features = teacher_outputs_full_det.get('det_intermediate_feature')
                if distill_weight_det_logits > 0: loss_distill_d_logits_val = criterion_det_logits_distill(student_d_outputs_logits, teacher_d_outputs_logits)
                if distill_weight_det_feature > 0 and student_d_inter_features is not None and teacher_d_inter_features is not None:
                    loss_distill_d_feature_val = criterion_det_feature_distill(student_d_inter_features, teacher_d_inter_features)

            loss_distill_c_val = torch.tensor(0.0, device=device)
            if distill_weight_cls > 0 and teacher_cls_model and replay_cls_iter and (i % replay_freq == 0):
                try: replay_c_images, _ = next(replay_cls_iter)
                except StopIteration: replay_cls_iter = iter(replay_cls_loader); replay_c_images, _ = next(replay_cls_iter)
                replay_c_images = replay_c_images.to(device)
                with torch.no_grad(): teacher_c_logits = teacher_cls_model(replay_c_images)['cls']
                student_c_logits = model(replay_c_images)['cls']
                loss_distill_c_val = criterion_cls_distill(F.log_softmax(student_c_logits / T, dim=1), F.softmax(teacher_c_logits / T, dim=1)) * (T * T)
            
            total_batch_loss = loss_main + distill_weight_det_logits * loss_distill_d_logits_val + distill_weight_det_feature * loss_distill_d_feature_val + distill_weight_cls * loss_distill_c_val
        
        scaler.scale(total_batch_loss).backward(); scaler.step(optimizer); scaler.update()
        total_loss_main += loss_main.item(); total_loss_distill_d_logits += loss_distill_d_logits_val.item() if isinstance(loss_distill_d_logits_val, torch.Tensor) else loss_distill_d_logits_val; total_loss_distill_d_feature += loss_distill_d_feature_val.item() if isinstance(loss_distill_d_feature_val, torch.Tensor) else loss_distill_d_feature_val; total_loss_distill_c += loss_distill_c_val.item() if isinstance(loss_distill_c_val, torch.Tensor) else loss_distill_c_val
        progress_bar.set_postfix({'seg_L': f"{loss_main.item():.4f}", 'det_D_Lgt': f"{loss_distill_d_logits_val.item():.4f}", 'det_D_Feat': f"{loss_distill_d_feature_val.item():.4f}", 'cls_D': f"{loss_distill_c_val.item():.4f}"})
    
    avg_loss_main = total_loss_main / len(seg_dataloader) if len(seg_dataloader) > 0 else 0
    num_distill_batches_d = (len(seg_dataloader) // replay_freq if replay_freq > 0 and replay_det_loader and len(seg_dataloader) > 0 else 1); avg_loss_distill_d_logits = total_loss_distill_d_logits / max(1, num_distill_batches_d); avg_loss_distill_d_feature = total_loss_distill_d_feature / max(1, num_distill_batches_d)
    num_distill_batches_c = (len(seg_dataloader) // replay_freq if replay_freq > 0 and replay_cls_loader and len(seg_dataloader) > 0 else 1); avg_loss_distill_c = total_loss_distill_c / max(1, num_distill_batches_c)
    logger.log(f"{current_stage_name} (Seg Main) | Epoch: {epoch+1} | Avg Seg Loss: {avg_loss_main:.4f} | Avg DetLogitsDistill: {avg_loss_distill_d_logits:.4f} | Avg DetFeatDistill: {avg_loss_distill_d_feature:.4f} | Avg Cls Distill: {avg_loss_distill_c:.4f}"); return avg_loss_main

# train_one_epoch_cls_distill_seg_det (分類為主，蒸餾分割，蒸餾偵測logits+feature)
def train_one_epoch_cls_distill_seg_det(
    model, teacher_seg_model, teacher_det_model,
    cls_dataloader, replay_seg_dataloader, replay_det_dataloader,
    criterion_cls, criterion_seg_distill, 
    criterion_det_logits_distill, criterion_det_feature_distill, 
    optimizer, device, logger, epoch,
    distill_weight_seg, distill_weight_det_logits, distill_weight_det_feature, 
    replay_freq, T, scaler, current_stage_name="Stage"
):
    model.train()
    if teacher_seg_model: teacher_seg_model.eval()
    if teacher_det_model: teacher_det_model.eval()
    total_loss_main, total_loss_distill_s, total_loss_distill_d_logits, total_loss_distill_d_feature = 0.0, 0.0, 0.0, 0.0
    if replay_seg_dataloader: replay_seg_iter = iter(replay_seg_dataloader)
    else: replay_seg_iter = None
    if replay_det_dataloader: replay_det_iter = iter(replay_det_dataloader)
    else: replay_det_iter = None

    progress_bar = tqdm(cls_dataloader, desc=f"{current_stage_name} (Cls Main) Epoch {epoch+1} Train", leave=False)
    for i, (cls_images, cls_labels) in enumerate(progress_bar):
        cls_images, cls_labels = cls_images.to(device), cls_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',enabled=scaler.is_enabled()):
            outputs = model(cls_images, return_intermediate_feature_for_det=False)
            cls_logits = outputs['cls']
            loss_main = criterion_cls(cls_logits, cls_labels)
            
            loss_distill_s_val = torch.tensor(0.0, device=device)
            if distill_weight_seg > 0 and teacher_seg_model and replay_seg_iter and (i % replay_freq == 0):
                try: replay_s_images, _ = next(replay_seg_iter)
                except StopIteration: replay_seg_iter = iter(replay_seg_dataloader); replay_s_images, _ = next(replay_seg_iter)
                replay_s_images = replay_s_images.to(device)
                with torch.no_grad(): teacher_s_logits = teacher_seg_model(replay_s_images)['seg']
                student_s_logits = model(replay_s_images)['seg']
                loss_distill_s_val = criterion_seg_distill(F.log_softmax(student_s_logits/T,dim=1), F.softmax(teacher_s_logits/T,dim=1)) * (T*T)

            loss_distill_d_logits_val, loss_distill_d_feature_val = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if (distill_weight_det_logits > 0 or distill_weight_det_feature > 0) and teacher_det_model and replay_det_iter and (i % replay_freq == 0):
                try: replay_d_images_tuple, _ = next(replay_det_iter)
                except StopIteration: replay_det_iter = iter(replay_det_dataloader); replay_d_images_tuple, _ = next(replay_det_iter)
                replay_d_images = torch.stack(replay_d_images_tuple, dim=0).to(device)
                student_outputs_full_det = model(replay_d_images, return_intermediate_feature_for_det=True)
                student_d_outputs_logits = student_outputs_full_det['det']
                student_d_inter_features = student_outputs_full_det.get('det_intermediate_feature')
                with torch.no_grad():
                    teacher_outputs_full_det = teacher_det_model(replay_d_images, return_intermediate_feature_for_det=True)
                    teacher_d_outputs_logits = teacher_outputs_full_det['det']
                    teacher_d_inter_features = teacher_outputs_full_det.get('det_intermediate_feature')
                if distill_weight_det_logits > 0: loss_distill_d_logits_val = criterion_det_logits_distill(student_d_outputs_logits, teacher_d_outputs_logits)
                if distill_weight_det_feature > 0 and student_d_inter_features is not None and teacher_d_inter_features is not None:
                    loss_distill_d_feature_val = criterion_det_feature_distill(student_d_inter_features, teacher_d_inter_features)
            
            total_batch_loss = loss_main + distill_weight_seg * loss_distill_s_val + distill_weight_det_logits * loss_distill_d_logits_val + distill_weight_det_feature * loss_distill_d_feature_val
        
        scaler.scale(total_batch_loss).backward(); scaler.step(optimizer); scaler.update()
        total_loss_main += loss_main.item(); total_loss_distill_s += loss_distill_s_val.item() if isinstance(loss_distill_s_val, torch.Tensor) else loss_distill_s_val; total_loss_distill_d_logits += loss_distill_d_logits_val.item() if isinstance(loss_distill_d_logits_val, torch.Tensor) else loss_distill_d_logits_val; total_loss_distill_d_feature += loss_distill_d_feature_val.item() if isinstance(loss_distill_d_feature_val, torch.Tensor) else loss_distill_d_feature_val
        progress_bar.set_postfix({'cls_L': f"{loss_main.item():.4f}", 'seg_D': f"{loss_distill_s_val.item():.4f}", 'det_D_Lgt': f"{loss_distill_d_logits_val.item():.4f}", 'det_D_Feat': f"{loss_distill_d_feature_val.item():.4f}"})
        
    avg_loss_main = total_loss_main / len(cls_dataloader) if len(cls_dataloader) > 0 else 0
    num_distill_batches_s = (len(cls_dataloader) // replay_freq if replay_freq > 0 and replay_seg_dataloader and len(cls_dataloader) > 0 else 1); avg_loss_distill_s = total_loss_distill_s / max(1, num_distill_batches_s)
    num_distill_batches_d = (len(cls_dataloader) // replay_freq if replay_freq > 0 and replay_det_dataloader and len(cls_dataloader) > 0 else 1); avg_loss_distill_d_logits = total_loss_distill_d_logits / max(1, num_distill_batches_d); avg_loss_distill_d_feature = total_loss_distill_d_feature / max(1, num_distill_batches_d)
    logger.log(f"{current_stage_name} (Cls Main) | Epoch: {epoch+1} | Avg Cls Loss: {avg_loss_main:.4f} | Avg Seg Distill: {avg_loss_distill_s:.4f} | Avg DetLogitsDistill: {avg_loss_distill_d_logits:.4f} | Avg DetFeatDistill: {avg_loss_distill_d_feature:.4f}"); return avg_loss_main

# --- 輔助函數：為優化器設定差分學習率 ---
def get_optimizer_with_differential_lr(model, base_lr, weight_decay, lr_factors_config):
    params_to_optimize = []
    param_names_added = set()
    def add_params_with_lr(module_name_prefix, module, factor):
        lr_val = base_lr * factor
        for name, param in module.named_parameters():
            full_name = f"{module_name_prefix}.{name}" if module_name_prefix else name
            if param.requires_grad and full_name not in param_names_added:
                params_to_optimize.append({'params': param, 'lr': lr_val})
                param_names_added.add(full_name)
    
    # 按照鍵名設定學習率因子
    if hasattr(model, 'backbone_features'): add_params_with_lr('backbone_features', model.backbone_features, lr_factors_config.get('backbone', 1.0))
    if hasattr(model, 'neck_conv1'): add_params_with_lr('neck_conv1', model.neck_conv1, lr_factors_config.get('neck', 1.0))
    if hasattr(model, 'neck_conv2'): add_params_with_lr('neck_conv2', model.neck_conv2, lr_factors_config.get('neck', 1.0))
    
    if hasattr(model, 'head_shared_conv1'): add_params_with_lr('head_shared_conv1', model.head_shared_conv1, lr_factors_config.get('shared_head_early', 1.0))
    if hasattr(model, 'head_shared_conv2'): add_params_with_lr('head_shared_conv2', model.head_shared_conv2, lr_factors_config.get('shared_head_mid', 1.0))
    if hasattr(model, 'head_shared_conv3'): add_params_with_lr('head_shared_conv3', model.head_shared_conv3, lr_factors_config.get('shared_head_late', 1.0))

    if hasattr(model, 'segmentation_head'): add_params_with_lr('segmentation_head', model.segmentation_head, lr_factors_config.get('seg_head', 1.0))
    
    # 處理新的兩層偵測頭
    if hasattr(model, 'detection_proj1') and hasattr(model, 'detection_bn_proj') and hasattr(model, 'detection_relu_proj') and hasattr(model, 'detection_head_final'):
        det_head_factor = lr_factors_config.get('det_head', 1.0)
        add_params_with_lr('detection_proj1', model.detection_proj1, det_head_factor)
        add_params_with_lr('detection_bn_proj', model.detection_bn_proj, det_head_factor)
        add_params_with_lr('detection_head_final', model.detection_head_final, det_head_factor)
    elif hasattr(model, 'detection_head'): # 向下兼容舊的單層偵測頭
         add_params_with_lr('detection_head', model.detection_head, lr_factors_config.get('det_head', 1.0))

    if hasattr(model, 'classification_fc'): add_params_with_lr('classification_fc', model.classification_fc, lr_factors_config.get('cls_head', 1.0))
    
    # 確保所有參數都被加入
    for name, param in model.named_parameters():
        if param.requires_grad and name not in param_names_added:
            logger.log(f"Warning: Parameter {name} was not explicitly assigned a learning rate factor, using base_lr.")
            params_to_optimize.append({'params': param, 'lr': base_lr})
            param_names_added.add(name)
            
    return optim.AdamW(params_to_optimize, lr=base_lr, weight_decay=weight_decay)


# --- 主訓練迴圈 (單次 Pass 版本) ---
def main_training_loop():
    global logger 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")
    use_amp = device.type == 'cuda' 
    scaler = torch.GradScaler(enabled=use_amp)

    # --- 資料載入器設定 (已包含 ColorJitter 和 replay_cls_loader) ---
    # ... (與你兩 Pass 版本中的資料載入器設定相同)
    DATA_ROOT = 'DL_data/' 
    BATCH_SIZE_MAIN = 32
    BATCH_SIZE_REPLAY = 8
    NUM_WORKERS = 2 if device.type == 'cuda' else 0

    specific_transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    specific_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    imagenette_path = os.path.join(DATA_ROOT, 'imagenette-160')
    def apply_specific_after_sync(sync_transform, specific_transform):
        return lambda pil_img: specific_transform(sync_transform(pil_img)[0])

    train_imagenette_dataset = datasets.ImageFolder(root=os.path.join(imagenette_path, 'train'),
        transform=apply_specific_after_sync(SynchronizedTransform(TARGET_IMG_SIZE, is_train=True), specific_transforms_train))
    val_imagenette_dataset = datasets.ImageFolder(root=os.path.join(imagenette_path, 'val'),
        transform=apply_specific_after_sync(SynchronizedTransform(TARGET_IMG_SIZE, is_train=False), specific_transforms_val))
    train_cls_loader = DataLoader(train_imagenette_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_cls_loader = DataLoader(val_imagenette_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    replay_cls_loader = DataLoader(train_imagenette_dataset, batch_size=BATCH_SIZE_REPLAY, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    logger.log(f"Cls DataLoaders: train={len(train_cls_loader.dataset)}, val={len(val_cls_loader.dataset)}, replay={len(replay_cls_loader.dataset)}")
    
    coco_images_base = os.path.join(DATA_ROOT, 'mini_coco_det', 'images')
    coco_annotations_base = os.path.join(DATA_ROOT, 'mini_coco_det', 'annotations')
    train_coco_dataset = CocoDetectionDataset(image_dir=os.path.join(coco_images_base, 'train'), annotation_file=os.path.join(coco_annotations_base, 'instances_train.json'), 
                                            common_transforms=SynchronizedTransform(TARGET_IMG_SIZE,is_train=True), specific_transforms=specific_transforms_train)
    val_coco_dataset = CocoDetectionDataset(image_dir=os.path.join(coco_images_base, 'val'), annotation_file=os.path.join(coco_annotations_base, 'instances_val.json'), 
                                         common_transforms=SynchronizedTransform(TARGET_IMG_SIZE,is_train=False), specific_transforms=specific_transforms_val)
    def collate_fn_detection(batch): return tuple(zip(*batch))
    train_det_loader = DataLoader(train_coco_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_detection)
    val_det_loader = DataLoader(val_coco_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_detection)
    replay_det_loader = DataLoader(train_coco_dataset, batch_size=BATCH_SIZE_REPLAY, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_detection)
    logger.log(f"Det DataLoaders: train={len(train_det_loader.dataset)}, val={len(val_det_loader.dataset)}, replay={len(replay_det_loader.dataset)}")

    voc_seg_root = os.path.join(DATA_ROOT, 'mini_voc_seg')
    train_voc_dataset = VOCSegmentationDataset(root_dir=voc_seg_root, image_set_file='train240.txt', 
                                            common_transforms=SynchronizedTransform(TARGET_IMG_SIZE, is_train=True), specific_transforms=specific_transforms_train)
    val_voc_dataset = VOCSegmentationDataset(root_dir=voc_seg_root, image_set_file='val60.txt', 
                                         common_transforms=SynchronizedTransform(TARGET_IMG_SIZE, is_train=False), specific_transforms=specific_transforms_val)
    train_seg_loader = DataLoader(train_voc_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_seg_loader = DataLoader(val_voc_dataset, batch_size=BATCH_SIZE_MAIN, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    replay_seg_loader = DataLoader(train_voc_dataset, batch_size=BATCH_SIZE_REPLAY, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    logger.log(f"Seg DataLoaders: train={len(train_seg_loader.dataset)}, val={len(val_seg_loader.dataset)}, replay={len(replay_seg_loader.dataset)}")

    initial_model_state = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES).state_dict()

    # --- 超參數 (單 Pass 版本) ---
    epochs_s1, epochs_s2, epochs_s3 = 50, 70, 40 # 可以為偵測階段增加 Epochs

    lr_s1_base, lr_s3_base = 1e-4, 1e-4 
    lr_s2_base = 1e-4  # 偵測任務的初始學習率
    
    # 學習率調節因子 (保持你上一輪實驗成功的因子設定作為起點)
    factor_backbone_preserve = 0.005     
    factor_neck_preserve = 0.01          
    factor_shared_early_preserve = 0.01  
    factor_shared_mid_preserve = 0.05    
    factor_main_task_related_shared_late = 1.0 
    factor_main_task_head = 1.0              
    factor_distill_head = 0.2 # 你上次設為 0.2，可以保持或微調           

    # 蒸餾權重 (使用你上一輪實驗的較優設定)
    distill_weight_seg = 4.0 
    distill_weight_det_logits = 5.0     
    distill_weight_det_feature = 2.5    
    distill_weight_cls = 1.0 
    distill_temp = 2.0    
      
    criterion_detection = DETLoss().to(device)
    criterion_segmentation = SEGLoss().to(device) 
    criterion_classification = CLSLoss().to(device) 
    criterion_distill_kldiv = nn.KLDivLoss(reduction='mean', log_target=False).to(device) 
    criterion_distill_mseloss = nn.MSELoss(reduction='mean').to(device)
    criterion_feature_distill = nn.MSELoss(reduction='mean').to(device)

    evaluator = Evaluator(device)
      
    teacher_S_model = None
    teacher_D_model = None
    # 在單 Pass 中，teacher_C_model 不會被用到，因為 Cls 是最後一個任務

    # === STAGE 1: Segmentation Training ===
    stage_name_s1 = "S1 (Seg)"
    logger.log("\n" + "="*20 + f" {stage_name_s1} Training " + "="*20)
    model_s1 = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES)
    model_s1.load_state_dict(copy.deepcopy(initial_model_state)); model_s1.to(device)
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=lr_s1_base, weight_decay=1e-4)
    scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, 'max', patience=7, factor=0.75) # 增加 patience
    best_s1_mIoU = 0.0
    for epoch in range(epochs_s1):
        train_one_epoch_seg(model_s1, train_seg_loader, criterion_segmentation, optimizer_s1, device, logger, epoch, scaler, current_stage_name=stage_name_s1)
        _, mIoU_val = valid_one_epoch_seg(model_s1, val_seg_loader, criterion_segmentation, evaluator, device, logger, epoch, stage_name=f"{stage_name_s1} Val")
        current_lr = optimizer_s1.param_groups[0]['lr']
        logger.log(f"{stage_name_s1} | Epoch: {epoch+1} | Current LR: {current_lr:.1e}")
        scheduler_s1.step(mIoU_val)
        if mIoU_val > best_s1_mIoU:
            best_s1_mIoU = mIoU_val; torch.save(model_s1.state_dict(), "model_s1_best.pt")
            logger.log(f"{stage_name_s1}: New best mIoU: {best_s1_mIoU:.4f}, model saved.")
    logger.log(f"{stage_name_s1} Finished. mIoU_base: {best_s1_mIoU:.4f}")
    teacher_S_model = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES); teacher_S_model.load_state_dict(torch.load("model_s1_best.pt")); teacher_S_model.to(device).eval()

    # === STAGE 2: Detection Training (Main Task), Distill Segmentation ===
    stage_name_s2 = "S2 (Det)"
    logger.log("\n" + "="*20 + f" {stage_name_s2} Training " + "="*20)
    model_s2 = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES)
    model_s2.load_state_dict(torch.load("model_s1_best.pt")); model_s2.to(device)
    
    lr_factors_config_s2 = {
        'backbone': factor_backbone_preserve,
        'neck': factor_neck_preserve,
        'shared_head_early': factor_shared_early_preserve, 
        'shared_head_mid': factor_shared_mid_preserve, 
        'shared_head_late': factor_main_task_related_shared_late, 
        'det_head': factor_main_task_head,       
        'seg_head': factor_distill_head,        
        'cls_head': 0.001 
    }
    optimizer_s2 = get_optimizer_with_differential_lr(model_s2, lr_s2_base, 1e-4, lr_factors_config_s2)
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, 'max', patience=7, factor=0.5)
    best_s2_mAP = 0.0
    for epoch in range(epochs_s2):
        # 注意：調用 train_one_epoch_det_distill_seg_cls 時，最後幾個蒸餾參數對應 cls，這裡設為 None 或 0
        train_one_epoch_det_distill_seg_cls(
            model_s2, teacher_S_model, None, # No Cls teacher yet
            train_det_loader, replay_seg_loader, None, # No Cls replay yet
            criterion_detection, criterion_distill_kldiv, None, # No Cls distill criterion
            optimizer_s2, device, logger, epoch,
            distill_weight_seg, 0.0, # distill_weight_cls is 0
            replay_freq=1, T=distill_temp, scaler=scaler, current_stage_name=stage_name_s2
        )
        mAP_val = valid_one_epoch_det(model_s2, val_det_loader, evaluator, device, logger, epoch, stage_name=f"{stage_name_s2} Val")
        current_lr = optimizer_s2.param_groups[0]['lr'] 
        logger.log(f"{stage_name_s2} | Epoch: {epoch+1} | Current Base LR for Optimizer: {current_lr:.1e}")
        scheduler_s2.step(mAP_val)
        if mAP_val > best_s2_mAP:
            best_s2_mAP = mAP_val; torch.save(model_s2.state_dict(), "model_s2_best.pt")
            logger.log(f"{stage_name_s2}: New best mAP: {best_s2_mAP:.4f}, model saved.")
    logger.log(f"{stage_name_s2} Finished. mAP_base: {best_s2_mAP:.4f}")
    teacher_D_model = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES); teacher_D_model.load_state_dict(torch.load("model_s2_best.pt")); teacher_D_model.to(device).eval()

    # === STAGE 3: Classification Training (Main Task), Distill Seg & Det (Logits + Feature) ===
    stage_name_s3 = "S3 (Cls)"
    logger.log("\n" + "="*20 + f" {stage_name_s3} Training " + "="*20)
    model_s3 = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES)
    model_s3.load_state_dict(torch.load("model_s2_best.pt")); model_s3.to(device)
    
    lr_factors_config_s3 = {
        'backbone': factor_backbone_preserve * 0.1, # 最後階段，更強力保護
        'neck': factor_neck_preserve * 0.5,
        'shared_head_early': factor_shared_early_preserve * 0.5, 
        'shared_head_mid': factor_shared_mid_preserve, # 可以考慮也降低一點
        'shared_head_late': factor_main_task_related_shared_late, 
        'cls_head': factor_main_task_head,
        'seg_head': factor_distill_head,
        'det_head': factor_distill_head 
    }
    optimizer_s3 = get_optimizer_with_differential_lr(model_s3, lr_s3_base, 1e-4, lr_factors_config_s3)
    scheduler_s3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s3, 'max', patience=7, factor=0.5)
    best_s3_top1 = 0.0
    for epoch in range(epochs_s3):
        train_one_epoch_cls_distill_seg_det(
            model_s3, teacher_S_model, teacher_D_model,
            train_cls_loader, replay_seg_loader, replay_det_loader,
            criterion_classification, criterion_distill_kldiv, 
            criterion_distill_mseloss, criterion_feature_distill, 
            optimizer_s3, device, logger, epoch,
            distill_weight_seg, distill_weight_det_logits, distill_weight_det_feature, 
            replay_freq=1, T=distill_temp, scaler=scaler, current_stage_name=stage_name_s3
        )
        top1_acc_val = valid_one_epoch_cls(model_s3, val_cls_loader, evaluator, device, logger, epoch, stage_name=f"{stage_name_s3} Val")
        current_lr = optimizer_s3.param_groups[-1]['lr'] 
        logger.log(f"{stage_name_s3} | Epoch: {epoch+1} | Current Main LR: {current_lr:.1e}")
        scheduler_s3.step(top1_acc_val)
        if top1_acc_val > best_s3_top1:
            best_s3_top1 = top1_acc_val; torch.save(model_s3.state_dict(), "model_final_best_1pass.pt") 
            logger.log(f"{stage_name_s3}: New best Top1: {best_s3_top1:.4f}, model saved.")
    logger.log(f"{stage_name_s3} Finished. Top1_final: {best_s3_top1:.4f}")
      
    logger.log("\n" + "="*30 + " SINGLE PASS TRAINING COMPLETE " + "="*30)
    logger.log("Final best model from single pass saved as model_final_best_1pass.pt")

    # --- 最終評估 (單 Pass) ---
    logger.log("\n" + "="*30 + " FINAL EVALUATION AFTER SINGLE PASS " + "="*30)
    final_model = UnifiedMultiTaskModel(NUM_DET_CLASSES, NUM_SEG_CLASSES, NUM_CLS_CLASSES)
    final_model.load_state_dict(torch.load("model_final_best_1pass.pt"))
    final_model.to(device).eval()

    _, final_mIoU = valid_one_epoch_seg(final_model, val_seg_loader, criterion_segmentation, evaluator, device, logger, -1, stage_name="Final Seg Eval")
    final_mAP = valid_one_epoch_det(final_model, val_det_loader, evaluator, device, logger, -1, stage_name="Final Det Eval")
    final_Top1 = valid_one_epoch_cls(final_model, val_cls_loader, evaluator, device, logger, -1, stage_name="Final Cls Eval")

    # 基線是在各自階段訓練完成後的最佳表現
    mIoU_base_final = best_s1_mIoU 
    mAP_base_final = best_s2_mAP  
    Top1_base_final = best_s3_top1 # 分類是最後一個任務，其 base 就是它本身的最佳值

    mIoU_drop_final_percent = 0.0
    if mIoU_base_final > 1e-5 : 
        mIoU_drop_final_percent = (mIoU_base_final - final_mIoU) / mIoU_base_final * 100
      
    mAP_drop_final_percent = 0.0
    if mAP_base_final > 1e-5 : 
        mAP_drop_final_percent = (mAP_base_final - final_mAP) / mAP_base_final * 100
      
    # 分類任務是最後訓練的，所以它的 drop 相對於它自身的高峰通常是0或負數
    Top1_drop_final_percent = 0.0
    if Top1_base_final > 1e-5: # 這裡的 Top1_base_final 是 stage 3 的 best_s3_top1
        Top1_drop_final_percent = (Top1_base_final - final_Top1) / Top1_base_final * 100


    logger.log(f"Final Metrics after Single Pass:")
    logger.log(f"  Base mIoU (S1): {mIoU_base_final:.4f}, Final mIoU (S3 model): {final_mIoU:.4f}, Drop: {mIoU_drop_final_percent:.2f}%")
    logger.log(f"  Base mAP (S2): {mAP_base_final:.4f}, Final mAP (S3 model): {final_mAP:.4f}, Drop: {mAP_drop_final_percent:.2f}%")
    logger.log(f"  Base Top1 (S3): {Top1_base_final:.4f}, Final Top1 (S3 model): {final_Top1:.4f}, Drop: {Top1_drop_final_percent:.2f}%")
      
    target_drop_percentage = 5.0
    logger.log(f"  Target Drop for Seg/Det/Cls <= {target_drop_percentage:.2f}%")

    seg_met = abs(mIoU_drop_final_percent) <= target_drop_percentage or mIoU_drop_final_percent < 0 
    det_met = abs(mAP_drop_final_percent) <= target_drop_percentage or mAP_drop_final_percent < 0
    cls_met = abs(Top1_drop_final_percent) <= target_drop_percentage or Top1_drop_final_percent < 0 # 分類通常不會遺忘自己
      
    logger.log(f"  Seg forgetting criterion {'MET' if seg_met else 'NOT MET'}.")
    logger.log(f"  Det forgetting criterion {'MET' if det_met else 'NOT MET'}.")
    logger.log(f"  Cls forgetting criterion {'MET' if cls_met else 'NOT MET'}.")


if __name__ == '__main__':
    main_training_loop()
    print("單次 Pass 訓練流程已執行。請檢查日誌。")