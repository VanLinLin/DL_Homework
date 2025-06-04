# models.py (確保是這個版本，增加了返回中間特徵的功能)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

class UnifiedMultiTaskModel(nn.Module):
    def __init__(self, num_detection_classes=10, num_segmentation_classes=20, num_classification_classes=10):
        super().__init__()

        self.num_detection_classes = num_detection_classes
        self.num_segmentation_classes = num_segmentation_classes
        self.num_classification_classes = num_classification_classes

        # 1. 骨幹網路 (Backbone)
        efficientnet_b0_full = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone_features = efficientnet_b0_full.features # 正確引用
        
        backbone_s16_out_channels = 112 
        backbone_s32_out_channels = 1280 

        # 2. 頸部 (Neck) - 融合 s16 和 s32 特徵
        self.neck_s32_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        neck_fusion_in_channels = backbone_s16_out_channels + backbone_s32_out_channels
        neck_intermediate_channels = 512
        neck_out_channels_final = 256

        self.neck_conv1 = nn.Sequential(
            nn.Conv2d(neck_fusion_in_channels, neck_intermediate_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(neck_intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.neck_conv2 = nn.Sequential(
            nn.Conv2d(neck_intermediate_channels, neck_out_channels_final, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(neck_out_channels_final),
            nn.ReLU(inplace=True)
        )
        
        # 3. 統一頭部 (Unified Head)
        head_shared_channels = 256
        self.head_shared_conv1 = nn.Sequential(
            nn.Conv2d(neck_out_channels_final, head_shared_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_shared_channels),
            nn.ReLU(inplace=True)
        )
        self.head_shared_conv2 = nn.Sequential(
            nn.Conv2d(head_shared_channels, head_shared_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_shared_channels),
            nn.ReLU(inplace=True)
        )
        self.head_shared_conv3 = nn.Sequential( # 最後一個共享卷積層
            nn.Conv2d(head_shared_channels, head_shared_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_shared_channels),
            nn.ReLU(inplace=True)
        )
        
        # 任務特定輸出層
        self.segmentation_head = nn.Conv2d(head_shared_channels, self.num_segmentation_classes, kernel_size=1)
        detection_output_channels = 4 + 1 + self.num_detection_classes
        # --- MODIFIED: Deeper Detection Head ---
        intermediate_det_channels = 128 # 可調的中間通道數，128或256
        self.detection_proj1 = nn.Conv2d(head_shared_channels, intermediate_det_channels, kernel_size=1, bias=False)
        self.detection_bn_proj = nn.BatchNorm2d(intermediate_det_channels)
        self.detection_relu_proj = nn.ReLU(inplace=True)
        self.detection_head_final = nn.Conv2d(intermediate_det_channels, detection_output_channels, kernel_size=1)
        self.classification_gap = nn.AdaptiveAvgPool2d(1)
        self.classification_fc = nn.Linear(head_shared_channels, self.num_classification_classes)

    # MODIFIED: forward 方法增加 return_intermediate_feature_for_det 參數
    def forward(self, x: torch.Tensor, return_intermediate_feature_for_det:bool=False) -> dict[str, torch.Tensor]:
        input_H, input_W = x.shape[-2:]

        f_s16, f_s32 = None, None
        current_feat = x
        # 確保正確提取 EfficientNet-B0 的 features[5] (s16) 和 features[8] (s32, 最終卷積輸出)
        # features 的索引是 0-8
        # 0: stem
        # 1-7: MBConv stages (features[1]是stage0, features[2]是stage1 ... features[7]是stage6)
        # features[5] 是 MBConv stage 4 的輸出 (112 channels, H/16)
        # features[8] 是最後的 Conv1x1 (1280 channels, H/32)
        
        # 修正特徵提取邏輯以匹配 EfficientNetFeatures 的結構
        # self.backbone_features IS efficientnet_b0_full.features
        # features[0] (stem)
        # features[1] (stage 0 MBConv) -> 16ch, H/2 (不使用)
        # features[2] (stage 1 MBConv) -> 24ch, H/4
        # features[3] (stage 2 MBConv) -> 40ch, H/8
        # features[4] (stage 3 MBConv) -> 80ch, H/16
        # features[5] (stage 4 MBConv) -> 112ch, H/16 <- 這是我們的 f_s16
        # features[6] (stage 5 MBConv) -> 192ch, H/32
        # features[7] (stage 6 MBConv) -> 320ch, H/32
        # features[8] (final 1x1 conv) -> 1280ch, H/32 <- 這是我們的 f_s32

        # 更直接的提取方式：
        temp_feat = self.backbone_features[0](x)    # Stem
        temp_feat = self.backbone_features[1](temp_feat) # Stage 0
        temp_feat = self.backbone_features[2](temp_feat) # Stage 1
        temp_feat = self.backbone_features[3](temp_feat) # Stage 2
        temp_feat = self.backbone_features[4](temp_feat) # Stage 3
        f_s16 = self.backbone_features[5](temp_feat)     # Stage 4 -> f_s16 (112ch, H/16)

        temp_feat_for_s32 = self.backbone_features[6](f_s16) # Stage 5
        temp_feat_for_s32 = self.backbone_features[7](temp_feat_for_s32) # Stage 6
        f_s32 = self.backbone_features[8](temp_feat_for_s32) # Final 1x1 Conv -> f_s32 (1280ch, H/32)
        
        if f_s16 is None or f_s32 is None: # 理論上，按順序執行不會是 None
             raise RuntimeError("Failed to extract features from backbone for s16/s32 fusion.")

        feat_s32_upsampled = self.neck_s32_upsample(f_s32) 
        fused_s16_features = torch.cat([f_s16, feat_s32_upsampled], dim=1)
        
        neck_feat_intermediate = self.neck_conv1(fused_s16_features)
        neck_features_s16 = self.neck_conv2(neck_feat_intermediate)
        
        shared_feat_conv1 = self.head_shared_conv1(neck_features_s16)
        shared_feat_conv2 = self.head_shared_conv2(shared_feat_conv1)
        shared_feat_conv3 = self.head_shared_conv3(shared_feat_conv2) # 我們將用此層進行特徵蒸餾
        
        seg_logits_small = self.segmentation_head(shared_feat_conv3)
        seg_logits = F.interpolate(seg_logits_small, size=(input_H, input_W), mode='bilinear', align_corners=False)

        # --- MODIFIED: Forward for Deeper Detection Head ---
        det_inter_feat = self.detection_proj1(shared_feat_conv3)
        det_inter_feat = self.detection_bn_proj(det_inter_feat)
        det_inter_feat = self.detection_relu_proj(det_inter_feat)
        det_outputs = self.detection_head_final(det_inter_feat)
        # --- END MODIFICATION ---

        
        cls_pooled_features = self.classification_gap(shared_feat_conv3)
        cls_pooled_features = torch.flatten(cls_pooled_features, 1)
        cls_logits = self.classification_fc(cls_pooled_features)
        
        outputs = {
            'seg': seg_logits,
            'det': det_outputs,
            'cls': cls_logits
        }
        # 如果請求，返回用於偵測特徵蒸餾的中間特徵
        if return_intermediate_feature_for_det:
            outputs['det_intermediate_feature'] = shared_feat_conv3 
            
        return outputs

if __name__ == '__main__':
    NUM_DET_CLASSES_MAIN = 10
    NUM_SEG_CLASSES_MAIN = 20
    NUM_CLS_CLASSES_MAIN = 10

    model_test = UnifiedMultiTaskModel(
        num_detection_classes=NUM_DET_CLASSES_MAIN,
        num_segmentation_classes=NUM_SEG_CLASSES_MAIN, 
        num_classification_classes=NUM_CLS_CLASSES_MAIN
    )
    
    total_params_backbone_only = 0
    # EfficientNet-B0 features is a Sequential module. Sum params of its children.
    # Corrected way to sum params for backbone_features which is a Sequential
    total_params_backbone_only = sum(p.numel() for p in model_test.backbone_features.parameters() if p.requires_grad)
    print(f"骨幹網路 (EfficientNet-B0 features) 參數數量: {total_params_backbone_only:,} (~{total_params_backbone_only/1e6:.2f}M)")

    total_params_model = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f"模型總參數數量 (修改後): {total_params_model:,} (~{total_params_model/1e6:.2f}M)")

    batch_size_test = 2
    # 保持測試輸入尺寸與 train.py 中的 TARGET_IMG_SIZE 一致，以驗證正常操作下的 shape
    input_height_test = 224 
    input_width_test = 224
    dummy_input_test = torch.randn(batch_size_test, 3, input_height_test, input_width_test)

    print(f"Dummy input shape for torchinfo: {dummy_input_test.shape}")

    model_test.eval() 
    with torch.no_grad():
        try:
            print("Attempting a direct forward pass (without intermediate features)...")
            outputs_test_direct = model_test(dummy_input_test, return_intermediate_feature_for_det=False)
            print(f"  Direct seg_logits shape: {outputs_test_direct['seg'].shape}") 
            print(f"  Direct det_outputs shape: {outputs_test_direct['det'].shape}") 
            print(f"  Direct cls_logits shape: {outputs_test_direct['cls'].shape}")
            print("Direct forward pass successful.")

            print("\nAttempting a direct forward pass (WITH intermediate features for det)...")
            outputs_test_with_inter = model_test(dummy_input_test, return_intermediate_feature_for_det=True)
            if 'det_intermediate_feature' in outputs_test_with_inter:
                print(f"  Intermediate feature for det shape: {outputs_test_with_inter['det_intermediate_feature'].shape}")
                print("Direct forward pass with intermediate features successful.")
            else:
                print("Error: 'det_intermediate_feature' not found in model output when requested.")

        except Exception as e:
            print(f"Error during direct forward pass: {e}")

        print("\nRunning torchinfo.summary...")
        # torchinfo.summary 不直接支持額外的 forward 參數，所以它會以默認方式運行 forward
        summary(model_test, input_data=dummy_input_test, col_names=["input_size", "output_size", "num_params", "trainable"], depth=4) # depth 4 可能更詳細
        
    print("\n模型架構修改 (Neck + return_intermediate_feature) 測試完成。")