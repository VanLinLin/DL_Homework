#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py - 統一多任務模型評估腳本
"""

import warnings
# 過濾掉來自 transformers.utils.generic 的特定 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF_func # 確保 TF_func 被正確引用
from torchvision.ops import nms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import argparse
import math

# 從 torchmetrics 導入 (假設已安裝)
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from torchmetrics import JaccardIndex, Accuracy as TorchmetricsAccuracy
except ImportError:
    print("警告: torchmetrics 未安裝，mAP 和 mIoU 將無法準確計算。請執行: pip install torchmetrics")
    # Fallback stubs
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

from models import UnifiedMultiTaskModel # 從你的 models.py 導入

# --- 全域常數 (應與 train.py 和 models.py 保持一致) ---
NUM_DET_CLASSES = 10
NUM_SEG_CLASSES = 20
NUM_CLS_CLASSES = 10
IGNORE_INDEX_SEG = -100 
TARGET_IMG_SIZE = 224
GRID_SIZE_S = TARGET_IMG_SIZE // 16 # 假設骨幹+頸部總步長為 16
BOXES_PER_CELL_B = 1

# --- Dataset 類別定義 (從 train.py 複製並稍作調整) ---
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
        except FileNotFoundError: 
            print(f"警告: 圖片未找到 {img_path}")
            return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), {'boxes': torch.zeros((0,4), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.long)}

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
        except FileNotFoundError: 
            print(f"警告: 圖片列表檔案未找到 {image_set_path}")
            self.image_filenames = []

    def __getitem__(self, idx):
        if not self.image_filenames: return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), torch.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE), dtype=torch.long)
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except FileNotFoundError: 
            print(f"警告: 圖片或遮罩未找到 {img_path} 或 {mask_path}")
            return torch.zeros((3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)), torch.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE), dtype=torch.long)
        
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

class SynchronizedTransform: # 評估時通常 is_train=False
    def __init__(self, size, is_train=False): # 評估時默認不進行訓練時的增強 (如翻轉)
        self.size = (size, size) if isinstance(size, int) else size
        self.is_train = is_train # 評估時應設為 False

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
        if self.is_train and torch.rand(1) < 0.5: # 評估時通常不執行
            image_pil_res = TF_func.hflip(image_pil_res)
            if isinstance(target_res, dict) and 'boxes' in target_res: 
                boxes = target_res['boxes']
                if boxes.numel() > 0:
                    orig_x1 = boxes[:, 0].clone(); boxes[:, 0] = self.size[0] - boxes[:, 2]; boxes[:, 2] = self.size[0] - orig_x1
                target_res['boxes'] = boxes
            elif isinstance(target_res, Image.Image): target_res = TF_func.hflip(target_res)
        return image_pil_res, target_res

# --- NMS 和評估器輔助函數 (從 train.py 複製) ---
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

def remap_seg_targets_for_metric(targets, ignore_index_metric=IGNORE_INDEX_SEG, num_actual_classes=NUM_SEG_CLASSES):
    new=targets.clone().long();mask_fg=(new>=1)&(new<=20);new[mask_fg]=new[mask_fg]-1;new[targets==0]=ignore_index_metric;new[targets==255]=ignore_index_metric;return new

class Evaluator:
    def __init__(self, device, num_seg_classes=NUM_SEG_CLASSES, num_cls_classes=NUM_CLS_CLASSES, num_det_classes=NUM_DET_CLASSES, seg_ignore_index=IGNORE_INDEX_SEG):
        self.device = device
        self.num_seg_classes = num_seg_classes 
        # --- MODIFIED: 加入 warn_on_many_detections=False ---
        self.det_metric = MeanAveragePrecision(
            box_format="xywh", 
            iou_type="bbox", 
            class_metrics=False
        ).to(device)
        # --- END MODIFICATION ---


        # --- MODIFIED: 在初始化後設定屬性以禁用警告 ---
        if hasattr(self.det_metric, 'warn_on_many_detections'):
            self.det_metric.warn_on_many_detections = False
        # --- END MODIFICATION ---

        self.seg_metric = JaccardIndex(task="multiclass", num_classes=self.num_seg_classes, ignore_index=seg_ignore_index, average='macro').to(device)
        self.cls_metric = TorchmetricsAccuracy(task="multiclass", num_classes=num_cls_classes, top_k=1).to(device)
        self.reset() # reset 方法現在應該分別 reset 每個 metric
    
    def reset(self): # 確保 reset 正確
        self.det_metric.reset()
        self.seg_metric.reset()
        self.cls_metric.reset()
        # self.results = {} # 如果 evaluator.compute 不再被統一調用，這個可能不需要
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
        if task=='det':
            # 注意：preds_model_output 應該是原始的模型輸出張量，NMS 在 Evaluator 內部或之前完成
            # yolo_v1_NMS 返回的是已經 NMS 後的列表，適合 torchmetrics
            decoded_preds_nms = yolo_v1_NMS(preds_model_output.cpu()) # 直接用 cpu() 後的張量
            preds_xywh = self.xyxy_to_xywh_list(decoded_preds_nms) 
            targets_xywh = self.xyxy_to_xywh_list(targets) # targets 是 DataLoader 輸出的原始標註列表
            self.det_metric.update(preds_xywh, targets_xywh)
        elif task=='seg':
            preds_logits=preds_model_output;pred_labels=torch.argmax(preds_logits,dim=1)
            remapped_target_labels=remap_seg_targets_for_metric(targets)
            self.seg_metric.update(pred_labels.to(self.device),remapped_target_labels.to(self.device))
        elif task=='cls':
            cls_logits=preds_model_output
            self.cls_metric.update(cls_logits.to(self.device),targets.to(self.device))
    def compute(self): 
        results = {}
        # 這些 try-except 仍然有用，以防某些 metric 未初始化或計算失敗
        try:
            det_res = self.det_metric.compute()
            results['mAP'] = det_res.get('map', torch.tensor(0.0)).item()
            results['mAP_50'] = det_res.get('map_50', torch.tensor(0.0)).item()
        except Exception: results['mAP'], results['mAP_50'] = 0.0, 0.0
        try: 
            results['mIoU'] = self.seg_metric.compute().item()
        except Exception: results['mIoU'] = 0.0
        try: 
            results['Top1'] = self.cls_metric.compute().item()
        except Exception: results['Top1'] = 0.0
        # self.results = results # 如果 evaluator.compute 不再被統一調用，這個可能不需要
        return results

# --- 評估單個任務的輔助函數 ---
def evaluate_segmentation(model, dataloader, evaluator, device):
    model.eval()
    evaluator.seg_metric.reset() # 只重置分割指標
    # total_loss 和 criterion 在評估時通常不是必須的，除非你要打印驗證損失
    # progress_bar = tqdm(dataloader, desc="Evaluating Segmentation", leave=False) # tqdm 保持不變
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating Segmentation", leave=False): # 直接在迴圈使用tqdm
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                seg_logits = outputs['seg']
            # evaluator.update(seg_logits.cpu(), masks.cpu(), task='seg') # 原來的update會更新evaluator內部狀態
            # 直接更新特定的 metric
            pred_labels = torch.argmax(seg_logits, dim=1)
            remapped_target_labels = remap_seg_targets_for_metric(masks, 
                                                                 ignore_index_metric=evaluator.seg_metric.ignore_index, 
                                                                 num_actual_classes=evaluator.num_seg_classes)
            evaluator.seg_metric.update(pred_labels.to(device), remapped_target_labels.to(device))
            
    # metrics = evaluator.compute() # 不再計算所有指標
    # mIoU_val = metrics.get('mIoU', 0.0)
    mIoU_val = evaluator.seg_metric.compute().item() # 只計算並獲取分割指標
    # print(f"Segmentation mIoU: {mIoU_val:.4f}") # 這一行移到 main 函數中統一打印
    return mIoU_val

def evaluate_detection(model, dataloader, evaluator, device):
    model.eval()
    evaluator.det_metric.reset() # 只重置偵測指標
    # progress_bar = tqdm(dataloader, desc="Evaluating Detection", leave=False)
    with torch.no_grad():
        for images_tuple, targets_list in tqdm(dataloader, desc="Evaluating Detection", leave=False):
            images = torch.stack(images_tuple, dim=0).to(device)
            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                det_preds_raw = outputs['det']
            
            # evaluator.update(det_preds_raw.cpu(), targets_list, task='det')
            # 直接更新特定的 metric
            decoded_preds_nms = yolo_v1_NMS(det_preds_raw.cpu()) 
            preds_xywh = evaluator.xyxy_to_xywh_list(decoded_preds_nms) 
            targets_xywh = evaluator.xyxy_to_xywh_list(targets_list)         
            evaluator.det_metric.update(preds_xywh, targets_xywh)

    # metrics = evaluator.compute()
    # mAP_val = metrics.get('mAP_50', 0.0)
    det_res = evaluator.det_metric.compute() # 只計算並獲取偵測指標
    mAP_val = det_res.get('map_50', torch.tensor(0.0)).item()
    # print(f"Detection mAP@0.5: {mAP_val:.4f}")
    return mAP_val

def evaluate_classification(model, dataloader, evaluator, device):
    model.eval()
    evaluator.cls_metric.reset() # 只重置分類指標
    # progress_bar = tqdm(dataloader, desc="Evaluating Classification", leave=False)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating Classification", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                cls_logits = outputs['cls']
            # evaluator.update(cls_logits.cpu(), labels.cpu(), task='cls')
            # 直接更新特定的 metric
            evaluator.cls_metric.update(cls_logits.to(device), labels.to(device))
            
    # metrics = evaluator.compute()
    # top1_acc = metrics.get('Top1', 0.0)
    top1_acc = evaluator.cls_metric.compute().item() # 只計算並獲取分類指標
    # print(f"Classification Top-1 Accuracy: {top1_acc:.4f}")
    return top1_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Unified Multi-Task Model.")
    parser.add_argument('--weights', type=str, default='model_final_best_1pass.pt', help="Path to the model weights (.pt file)")
    parser.add_argument('--data_root', type=str, default='DL_data/', help="Root directory of the datasets")
    parser.add_argument('--tasks', type=str, default='all', choices=['all', 'seg', 'det', 'cls'], nargs='+',
                        help="Tasks to evaluate: 'all', or a space-separated list like 'seg det cls'")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers")
    
    # --- 新增：接收基線指標的參數 ---
    parser.add_argument('--base_miou', type=float, default=0.5594, help="Baseline mIoU for Segmentation from training_log_complete.txt")
    parser.add_argument('--base_map', type=float, default=0.0199, help="Baseline mAP@0.5 for Detection from training_log_complete.txt")
    parser.add_argument('--base_top1', type=float, default=0.7167, help="Baseline Top-1 Accuracy for Classification from training_log_complete.txt")
    # --- 結束新增 ---
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UnifiedMultiTaskModel(
        num_detection_classes=NUM_DET_CLASSES,
        num_segmentation_classes=NUM_SEG_CLASSES,
        num_classification_classes=NUM_CLS_CLASSES
    )
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"模型權重從 {args.weights} 載入成功。")
    except Exception as e:
        print(f"錯誤：無法載入模型權重 {args.weights}。錯誤訊息: {e}")
        return
    model.to(device)
    model.eval()

    evaluator = Evaluator(device) # 確保 Evaluator 初始化正確

    # --- Transforms (與你 eval.py 中一致) ---
    specific_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    common_transforms_val = SynchronizedTransform(TARGET_IMG_SIZE, is_train=False)
    def apply_specific_after_sync_val(sync_transform, specific_transform):
        return lambda pil_img: specific_transform(sync_transform(pil_img)[0])

    # --- 資料載入器 (與你 eval.py 中一致) ---
    val_seg_loader, val_det_loader, val_cls_loader = None, None, None
    if 'all' in args.tasks or 'seg' in args.tasks:
        voc_seg_root = os.path.join(args.data_root, 'mini_voc_seg')
        val_voc_dataset = VOCSegmentationDataset(root_dir=voc_seg_root, image_set_file='val60.txt', common_transforms=common_transforms_val, specific_transforms=specific_transforms_val)
        val_seg_loader = DataLoader(val_voc_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if not val_voc_dataset.image_filenames: print(f"警告: 分割驗證集為空或未載入。路徑: {voc_seg_root}")
    if 'all' in args.tasks or 'det' in args.tasks:
        coco_images_val = os.path.join(args.data_root, 'mini_coco_det', 'images', 'val')
        coco_annotations_val = os.path.join(args.data_root, 'mini_coco_det', 'annotations', 'instances_val.json')
        val_coco_dataset = CocoDetectionDataset(image_dir=coco_images_val, annotation_file=coco_annotations_val, common_transforms=common_transforms_val, specific_transforms=specific_transforms_val)
        def collate_fn_detection(batch): return tuple(zip(*batch))
        val_det_loader = DataLoader(val_coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_detection)
        if not val_coco_dataset.images_log: print(f"警告: 偵測驗證集為空或未載入。路徑: {coco_images_val}")
    if 'all' in args.tasks or 'cls' in args.tasks:
        imagenette_val_path = os.path.join(args.data_root, 'imagenette-160', 'val')
        val_imagenette_dataset = datasets.ImageFolder(root=imagenette_val_path, transform=apply_specific_after_sync_val(SynchronizedTransform(TARGET_IMG_SIZE, is_train=False), specific_transforms_val))
        val_cls_loader = DataLoader(val_imagenette_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if not val_imagenette_dataset.samples: print(f"警告: 分類驗證集為空或未載入。路徑: {imagenette_val_path}")

    # --- 執行評估 ---
    final_mIoU, final_mAP, final_Top1 = None, None, None

    if val_seg_loader and ('all' in args.tasks or 'seg' in args.tasks):
        print("\n--- Evaluating Segmentation ---")
        final_mIoU = evaluate_segmentation(model, val_seg_loader, evaluator, device) # 假設 evaluate_segmentation 返回 mIoU
        print(f"Segmentation mIoU: {final_mIoU:.4f}")

    if val_det_loader and ('all' in args.tasks or 'det' in args.tasks):
        print("\n--- Evaluating Detection ---")
        final_mAP = evaluate_detection(model, val_det_loader, evaluator, device) # 假設 evaluate_detection 返回 mAP50
        print(f"Detection mAP@0.5: {final_mAP:.4f}")

    if val_cls_loader and ('all' in args.tasks or 'cls' in args.tasks):
        print("\n--- Evaluating Classification ---")
        final_Top1 = evaluate_classification(model, val_cls_loader, evaluator, device) # 假設 evaluate_classification 返回 Top1
        print(f"Classification Top-1 Accuracy: {final_Top1:.4f}")
    
    # --- MODIFIED: 美化最終評估總結的輸出 ---
    print("\n" + "="*60)
    print(f"{'FINAL EVALUATION SUMMARY':^60}")
    print("="*60)
    print(f"Evaluated Model: {args.weights}\n")

    header = f"{'Task':<15} | {'Baseline':<10} | {'Final':<10} | {'Drop (%)':<10} | {'Status':<6}"
    print(header)
    print("-" * len(header))

    target_drop_percentage = 5.0
    
    if final_mIoU is not None and args.base_miou is not None:
        mIoU_drop_percent = 0.0
        if abs(args.base_miou) > 1e-6: # 避免除以零
            mIoU_drop_percent = (args.base_miou - final_mIoU) / args.base_miou * 100
        seg_met_str = "MET" if abs(mIoU_drop_percent) <= target_drop_percentage or mIoU_drop_percent < 0 else "NOT MET"
        print(f"{'Seg (mIoU)':<15} | {args.base_miou:<10.4f} | {final_mIoU:<10.4f} | {mIoU_drop_percent:<9.2f}% | {seg_met_str:<6}")
    elif final_mIoU is not None:
        print(f"{'Seg (mIoU)':<15} | {'N/A':<10} | {final_mIoU:<10.4f} | {'N/A':<10} | {'N/A':<6}")

    if final_mAP is not None and args.base_map is not None:
        mAP_drop_percent = 0.0
        if abs(args.base_map) > 1e-6:
            mAP_drop_percent = (args.base_map - final_mAP) / args.base_map * 100
        det_met_str = "MET" if abs(mAP_drop_percent) <= target_drop_percentage or mAP_drop_percent < 0 else "NOT MET"
        print(f"{'Det (mAP@.5)':<15} | {args.base_map:<10.4f} | {final_mAP:<10.4f} | {mAP_drop_percent:<9.2f}% | {det_met_str:<6}")
    elif final_mAP is not None:
        print(f"{'Det (mAP@.5)':<15} | {'N/A':<10} | {final_mAP:<10.4f} | {'N/A':<10} | {'N/A':<6}")

    if final_Top1 is not None and args.base_top1 is not None:
        Top1_drop_percent = 0.0
        if abs(args.base_top1) > 1e-6:
            Top1_drop_percent = (args.base_top1 - final_Top1) / args.base_top1 * 100
        cls_met_str = "MET" if abs(Top1_drop_percent) <= target_drop_percentage or Top1_drop_percent < 0 else "NOT MET"
        print(f"{'Cls (Top-1)':<15} | {args.base_top1:<10.4f} | {final_Top1:<10.4f} | {Top1_drop_percent:<9.2f}% | {cls_met_str:<6}")
    elif final_Top1 is not None:
        print(f"{'Cls (Top-1)':<15} | {'N/A':<10} | {final_Top1:<10.4f} | {'N/A':<10} | {'N/A':<6}")
    
    print("-" * len(header))
    print(f"Target Drop for each task considered MET if <= {target_drop_percentage:.2f}% (or if performance improved).")
    print("="*60)
    # --- END MODIFICATION ---

if __name__ == '__main__':
    main()
