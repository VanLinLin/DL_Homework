from torchvision import datasets, transforms
import os
import torch
from torch.utils.data import Dataset, DataLoader # DataLoader 新增
from PIL import Image
import json
import numpy as np # 新增 numpy 用於遮罩轉換

# --- 你原本的程式碼開始 ---
data_root = 'DL_data/' # 確保這個路徑是你的資料集根目錄
imagenette_path = os.path.join(data_root, 'imagenette-160')

# 影像轉換流程 (ImageNet 標準)
imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 範例尺寸，需根據模型調整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 創建 Dataset (Imagenette)
try:
    train_imagenette_dataset = datasets.ImageFolder(
        root=os.path.join(imagenette_path, 'train'),
        transform=imagenet_transform
    )
    val_imagenette_dataset = datasets.ImageFolder(
        root=os.path.join(imagenette_path, 'val'),
        transform=imagenet_transform
    )
    # 取得分類任務的類別數量
    NUM_CLASSIFICATION_CLASSES = len(train_imagenette_dataset.classes)
except FileNotFoundError:
    print(f"警告: Imagenette 資料集路徑未找到: {imagenette_path}")
    train_imagenette_dataset = None
    val_imagenette_dataset = None
    NUM_CLASSIFICATION_CLASSES = 10 # 預設為 Imagenette 的 10 類


class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, common_transforms=None, specific_transforms=None): # 修改參數名
        self.image_dir = image_dir # 修改：直接指向圖片資料夾 train/ 或 val/
        self.common_transforms = common_transforms # 給圖片和標註共同使用的轉換 (例如隨機翻轉)
        self.specific_transforms = specific_transforms # 給圖片單獨使用的轉換 (例如 ToTensor, Normalize)

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images_info = self.coco_data['images']
        
        # 建立 image_id 到 annotations 的映射
        self.img_to_anns = {}
        if 'annotations' in self.coco_data:
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        # 建立 image_id 到 image_info 的映射 (方便查找檔案名稱)
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.images_info}

        # 處理類別 ID 映射
        self.cat_ids = []
        if 'categories' in self.coco_data:
            for cat in self.coco_data['categories']:
                self.cat_ids.append(cat['id'])
        # 排序並建立映射到連續 ID (0 到 num_classes-1)
        self.sorted_cat_ids = sorted(list(set(self.cat_ids)))
        self.cat_id_to_contiguous_id = {
            cat_id: i for i, cat_id in enumerate(self.sorted_cat_ids)
        }
        # 偵測任務的類別數量
        # NUM_DETECTION_CLASSES = len(self.sorted_cat_ids) # 作業指定10個類別 [cite: 105]
        # 實際上，Mini-COCO-Det的JSON檔案通常已經處理好類別ID。
        # 如果你的JSON中的category_id本身就是0-9，那就不需要複雜映射。
        # 這裡我們假設直接使用 JSON 中的 category_id，但實際應用時需確認。

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_id = img_info['id']
        img_filename = self.img_id_to_filename[img_id]
        
        img_path = os.path.join(self.image_dir, img_filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"錯誤：圖片檔案未找到 {img_path}")
            # 返回一個假的圖片和標註，或者拋出錯誤
            return torch.zeros((3, 224, 224)), {'boxes': torch.zeros((0,4)), 'labels': torch.zeros((0,), dtype=torch.long)}


        target = {'boxes': [], 'labels': []}
        original_size = image.size # (width, height)

        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                bbox = ann['bbox'] # [x, y, width, height]
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                
                # 進行邊界檢查，防止 bbox 超出影像範圍 (COCO 數據集通常是準確的)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(original_size[0], xmax)
                ymax = min(original_size[1], ymax)

                # 確保 w 和 h > 0
                if xmax > xmin and ymax > ymin:
                    target['boxes'].append([xmin, ymin, xmax, ymax])
                    # 假設 ann['category_id'] 已經是我們想要的索引 (例如 0-9 for 10 classes)
                    # 如果不是，你需要使用 self.cat_id_to_contiguous_id[ann['category_id']]
                    target['labels'].append(ann['category_id'] -1) # 假設COCO原始ID從1開始，我們要映射到0開始
                                                                 # 這部分你需要根據你的 JSON 檔案確認

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        if not target['boxes'].numel(): # 如果為空
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        # 轉換
        # 實際訓練中，這裡的轉換需要能同時處理影像和邊界框
        # 例如 albumentations 函式庫，或者 torchvision.transforms.v2 (如果可用)
        # 簡易測試：只對影像做基本轉換
        if self.common_transforms: # 假設 common_transforms 能處理 (image, target)
             image, target = self.common_transforms(image, target)
        
        if self.specific_transforms: # 假設 specific_transforms 只處理 image
            image = self.specific_transforms(image)


        return image, target

    def __len__(self):
        return len(self.images_info)


class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set_file, common_transforms=None, specific_transforms=None): # 修改參數
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')
        self.common_transforms = common_transforms
        self.specific_transforms = specific_transforms


        image_set_path = os.path.join(root_dir, 'ImageSets', 'Segmentation', image_set_file)
        try:
            with open(image_set_path, 'r') as f:
                self.image_filenames = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"警告: VOC image set file 未找到: {image_set_path}")
            self.image_filenames = []


    def __getitem__(self, idx):
        if not self.image_filenames: # 如果檔案列表為空
            return torch.zeros((3,224,224)), torch.zeros((224,224), dtype=torch.long)

        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path) # 遮罩通常是 P 模式
        except FileNotFoundError:
            print(f"錯誤：檔案未找到 {img_path} 或 {mask_path}")
            return torch.zeros((3,224,224)), torch.zeros((224,224), dtype=torch.long)


        # 轉換 (非常重要：影像和遮罩需要同步轉換)
        # 實際訓練中，這裡的轉換需要能同時處理影像和遮罩
        # 簡易測試：
        if self.common_transforms: # 假設 common_transforms 能處理 (image, mask)
            image, mask = self.common_transforms(image, mask)

        if self.specific_transforms: # 假設 specific_transforms 只處理 image
            image = self.specific_transforms(image)
        
        # 將遮罩 PIL Image 轉換為 Tensor
        # PASCAL VOC 遮罩: 0-20 是類別, 255 是邊界/忽略.
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        # 處理忽略標籤 (例如將 255 映射到一個特定的忽略索引，或者在損失函數中忽略)
        # 這裡我們先不改變它，但在計算損失時要注意
        # mask_tensor[mask_tensor == 255] = IGNORE_INDEX # IGNORE_INDEX 通常是 -100 或 255

        return image, mask_tensor

    def __len__(self):
        return len(self.image_filenames)

# --- 你原本的程式碼結束 ---


# --- 測試區塊 ---
if __name__ == "__main__":
    print("開始測試 Dataset 類別...")

    # --- 1. 測試 Imagenette (分類) ---
    print("\n--- 測試 Imagenette Dataset ---")
    if train_imagenette_dataset:
        print(f"訓練集樣本數: {len(train_imagenette_dataset)}")
        print(f"驗證集樣本數: {len(val_imagenette_dataset)}")
        print(f"分類類別數量: {NUM_CLASSIFICATION_CLASSES}")
        print(f"類別名稱: {train_imagenette_dataset.classes}")
        img, label = train_imagenette_dataset[0]
        print(f"Imagenette 樣本 - 圖片 shape: {img.shape}, 圖片 dtype: {img.dtype}")
        print(f"Imagenette 樣本 - 標籤: {label}, 標籤 dtype: {type(label)}")
    else:
        print("Imagenette 資料集未載入，跳過測試。")

    # --- 2. 測試 Mini-COCO-Det (物件偵測) ---
    print("\n--- 測試 Mini-COCO-Det Dataset ---")
    coco_images_base = os.path.join(data_root, 'mini_coco_det', 'images')
    coco_annotations_base = os.path.join(data_root, 'mini_coco_det', 'annotations')

    def get_coco_transforms(is_train):
        img_specific_transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
        return None, img_specific_transforms

    _, coco_specific_train_transforms = get_coco_transforms(is_train=True)
    _, coco_specific_val_transforms = get_coco_transforms(is_train=False)

    try:
        train_coco_dataset = CocoDetectionDataset(
            image_dir=os.path.join(coco_images_base, 'train'),
            annotation_file=os.path.join(coco_annotations_base, 'instances_train.json'),
            specific_transforms=coco_specific_train_transforms
        )
        # val_coco_dataset = CocoDetectionDataset(...) # 類似地定義 val

        print(f"COCO 訓練集樣本數: {len(train_coco_dataset)}")
        if len(train_coco_dataset) > 0:
            img_coco, target_coco = train_coco_dataset[0]
            print(f"COCO 樣本 - 圖片 shape: {img_coco.shape}, 圖片 dtype: {img_coco.dtype}")
            print(f"COCO 樣本 - 標註 boxes shape: {target_coco['boxes'].shape}, labels shape: {target_coco['labels'].shape}")
            if target_coco['boxes'].numel() > 0:
                print(f"COCO 樣本 - 第一個 box: {target_coco['boxes'][0]}, label: {target_coco['labels'][0]}")
            else:
                print("COCO 樣本 - 此圖片無標註物件。")
        else:
            print("COCO 訓練集為空或載入失敗。")
    except FileNotFoundError as e:
        print(f"錯誤: COCO 資料集檔案未找到: {e}")
    except Exception as e:
        print(f"載入 COCO 資料集時發生錯誤: {e}")


    # --- 3. 測試 Mini-VOC-Seg (語義分割) ---
    print("\n--- 測試 Mini-VOC-Seg Dataset ---")
    voc_seg_root = os.path.join(data_root, 'mini_voc_seg')

    # 針對 VOC 的 specific_transforms (只用於影像)
    voc_specific_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 通常正規化也在此
    ])
    # 注意：common_transforms (同步轉換影像和遮罩) 在這裡仍然是 None。
    # 在實際訓練中，Resize 應該是 common_transform 的一部分，或者你需要確保 image 和 mask 都被 resize 到相同大小。

    try:
        train_voc_dataset = VOCSegmentationDataset(
            root_dir=voc_seg_root,
            image_set_file='train240.txt',
            common_transforms=None, # 明確設為 None
            specific_transforms=voc_specific_transforms # 這個會把影像 Resize 並轉為 Tensor
        )
        # val_voc_dataset = VOCSegmentationDataset(...) # 類似地定義 val

        print(f"VOC Seg 訓練集樣本數: {len(train_voc_dataset)}")
        if len(train_voc_dataset) > 0:
            # 從 dataset 取出時，img_voc 已經是 (C, 224, 224) 的 Tensor
            # mask_voc 是 (H_original, W_original) 的 LongTensor
            img_voc, mask_voc = train_voc_dataset[0]

            print(f"VOC Seg 樣本 - 圖片 shape: {img_voc.shape}, 圖片 dtype: {img_voc.dtype}") # 應該是 (C, 224, 224)
            print(f"VOC Seg 樣本 - 遮罩 (原始維度) shape: {mask_voc.shape}, 遮罩 (原始維度) dtype: {mask_voc.dtype}")

            # 手動將遮罩 Resize 到與圖片相同的 (224, 224) 以便檢查
            # transforms.Resize 作用於 Tensor 時，期望輸入是 (C, H, W) 或 (H, W)
            if mask_voc.ndim == 2: # (H, W)
                mask_resized = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)(mask_voc.unsqueeze(0)).squeeze(0)
            elif mask_voc.ndim == 1 and mask_voc.numel() == 0: # 空遮罩
                mask_resized = torch.zeros((224,224), dtype=torch.long) # 建立一個空的resize後遮罩
                print("VOC Seg 樣本 - 遮罩為空。")
            else:
                print(f"VOC Seg 樣本 - 未預期的遮罩維度: {mask_voc.shape}, 無法自動 resize。")
                mask_resized = mask_voc # 保持原樣或進行其他處理

            if mask_voc.numel() > 0 or mask_resized.numel() > 0 : # 只有在遮罩非空時才打印
                print(f"VOC Seg 樣本 - 遮罩 (Resize後) shape: {mask_resized.shape}, 遮罩 (Resize後) dtype: {mask_resized.dtype}")
                print(f"VOC Seg 樣本 - 遮罩 (Resize後) 唯一值: {torch.unique(mask_resized)}")
        else:
            print("VOC Seg 訓練集為空或載入失敗。")

    except FileNotFoundError as e:
        print(f"錯誤: VOC Seg 資料集檔案未找到: {e}")
    except Exception as e:
        print(f"載入 VOC Seg 資料集時發生錯誤: {e}")

    print("\n測試完成。")