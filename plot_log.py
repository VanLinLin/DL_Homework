import matplotlib
matplotlib.use('Agg') # <--- 新增：在導入 pyplot 前設定後端為 'Agg'
import matplotlib.pyplot as plt
import re
import os

def parse_log_file(log_file_path="training_log.txt"):
    print(f"開始解析日誌檔案: {log_file_path}") # 除錯信息
    metrics = {
        "s1_seg_train_loss": [], "s1_seg_val_miou": [], "s1_seg_val_loss": [],
        "s2_det_train_loss": [], "s2_det_val_map": [], "s2_seg_distill_loss": [],
        "s3_cls_train_loss": [], "s3_cls_val_top1": [], "s3_seg_distill_loss": [],
        "s3_det_logits_distill_loss": [], "s3_det_feature_distill_loss": []
    }
    
    regex_s1_train = re.compile(r"S1 \(Seg\) \(Seg\) \| Epoch: (\d+) \| Average Training Loss: (\d+\.\d+)")
    regex_s1_val = re.compile(r"S1 \(Seg\) Val \| Epoch: (\d+) \| Validation Loss: (\d+\.\d+) \| mIoU: (\d+\.\d+)")
    regex_s2_train = re.compile(r"S2 \(Det\) \(Det Main\) \| Epoch: (\d+) \| Avg Det Loss: (\d+\.\d+) \| Avg Seg Distill: (\d+\.\d+) \| Avg Cls Distill: (\d+\.\d+)")
    regex_s2_val = re.compile(r"S2 \(Det\) Val \| Epoch: (\d+) \| Validation mAP@0.5: (\d+\.\d+)")
    regex_s3_train = re.compile(r"S3 \(Cls\) \(Cls Main\) \| Epoch: (\d+) \| Avg Cls Loss: (\d+\.\d+) \| Avg Seg Distill: (\d+\.\d+) \| Avg DetLogitsDistill: (\d+\.\d+) \| Avg DetFeatDistill: (\d+\.\d+)")
    regex_s3_val = re.compile(r"S3 \(Cls\) Val \| Epoch: (\d+) \| Validation Top-1 Acc: (\d+\.\d+)")

    line_count = 0
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                line_count += 1
                if line_count % 50 == 0: # 每處理50行打印一次進度
                    print(f"  已解析 {line_count} 行...")
                
                match_s1_train = regex_s1_train.search(line)
                if match_s1_train:
                    metrics["s1_seg_train_loss"].append(float(match_s1_train.group(2)))
                    continue
                match_s1_val = regex_s1_val.search(line)
                if match_s1_val:
                    metrics["s1_seg_val_loss"].append(float(match_s1_val.group(2)))
                    metrics["s1_seg_val_miou"].append(float(match_s1_val.group(3)))
                    continue
                match_s2_train = regex_s2_train.search(line)
                if match_s2_train:
                    metrics["s2_det_train_loss"].append(float(match_s2_train.group(2)))
                    metrics["s2_seg_distill_loss"].append(float(match_s2_train.group(3)))
                    continue
                match_s2_val = regex_s2_val.search(line)
                if match_s2_val:
                    metrics["s2_det_val_map"].append(float(match_s2_val.group(2)))
                    continue
                match_s3_train = regex_s3_train.search(line)
                if match_s3_train:
                    metrics["s3_cls_train_loss"].append(float(match_s3_train.group(2)))
                    metrics["s3_seg_distill_loss"].append(float(match_s3_train.group(3)))
                    metrics["s3_det_logits_distill_loss"].append(float(match_s3_train.group(4)))
                    metrics["s3_det_feature_distill_loss"].append(float(match_s3_train.group(5)))
                    continue
                match_s3_val = regex_s3_val.search(line)
                if match_s3_val:
                    metrics["s3_cls_val_top1"].append(float(match_s3_val.group(2)))
                    continue
        print(f"日誌檔案解析完成，共 {line_count} 行。") # 除錯信息
    except FileNotFoundError:
        print(f"錯誤: 日誌檔案 '{log_file_path}' 未找到。")
        return None
    
    # 打印解析到的數據點數量，幫助檢查解析是否完整
    for key, value in metrics.items():
        print(f"  指標 '{key}': 提取到 {len(value)} 個數據點")
    return metrics

def plot_metrics(metrics_data, output_dir="plots"): # 假設此函數在你的 plot_log.py 中
    """根據解析的指標數據繪製圖表。"""
    if not metrics_data:
        print("沒有解析到數據，無法繪製圖表。")
        return

    print(f"開始繪製圖表並儲存到 '{output_dir}' 資料夾...")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"  已創建資料夾: {output_dir}")
        except OSError as e:
            print(f"  錯誤: 無法創建資料夾 {output_dir} - {e}")
            return

    # Plot Stage 1: Segmentation
    if metrics_data["s1_seg_train_loss"] and metrics_data["s1_seg_val_miou"]:
        print("  正在繪製 Stage 1 圖表...")
        epochs_s1 = range(1, len(metrics_data["s1_seg_train_loss"]) + 1)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss (Seg)', color=color)
        ax1.plot(epochs_s1, metrics_data["s1_seg_train_loss"], color=color, linestyle='-', label='S1 Train Loss')
        if metrics_data["s1_seg_val_loss"]:
             ax1.plot(epochs_s1, metrics_data["s1_seg_val_loss"], color=color, linestyle=':', label='S1 Val Loss')
        ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='upper left')
        ax2 = ax1.twinx(); color = 'tab:blue'
        ax2.set_ylabel('Validation mIoU (Seg)', color=color)
        ax2.plot(epochs_s1, metrics_data["s1_seg_val_miou"], color=color, linestyle='-', marker='o', label='S1 Val mIoU')
        ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='upper right')
        
        plt.title('Stage 1: Segmentation Training Metrics') # 標題
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.90) # MODIFIED: 調整子圖邊距
        # 或者使用 fig.tight_layout(pad=1.5) 或 plt.title('...', pad=20)

        try:
            save_path = os.path.join(output_dir, "stage1_segmentation_metrics.png")
            plt.savefig(save_path)
            print(f"  Stage 1 圖表已儲存到: {save_path}")
        except Exception as e:
            print(f"  錯誤: 儲存 Stage 1 圖表失敗 - {e}")
        plt.close(fig)
    else:
        print("  警告: Stage 1 數據不足，跳過繪圖。")

    # Plot Stage 2: Detection
    if metrics_data["s2_det_train_loss"] and metrics_data["s2_det_val_map"]:
        print("  正在繪製 Stage 2 圖表...")
        epochs_s2 = range(1, len(metrics_data["s2_det_train_loss"]) + 1)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Losses (Det)', color=color)
        ax1.plot(epochs_s2, metrics_data["s2_det_train_loss"], color=color, linestyle='-', label='S2 Det Train Loss')
        if metrics_data["s2_seg_distill_loss"]: 
            ax1.plot(epochs_s2, metrics_data["s2_seg_distill_loss"], color='tab:orange', linestyle=':', label='S2 Seg Distill Loss')
        ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='upper left')
        ax2 = ax1.twinx(); color = 'tab:blue'
        ax2.set_ylabel('Validation mAP@0.5 (Det)', color=color)
        ax2.plot(epochs_s2, metrics_data["s2_det_val_map"], color=color, linestyle='-', marker='o', label='S2 Val mAP@0.5')
        ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='upper right')

        plt.title('Stage 2: Detection Training Metrics') # 標題
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.90) # MODIFIED: 調整子圖邊距

        try:
            save_path = os.path.join(output_dir, "stage2_detection_metrics.png")
            plt.savefig(save_path)
            print(f"  Stage 2 圖表已儲存到: {save_path}")
        except Exception as e:
            print(f"  錯誤: 儲存 Stage 2 圖表失敗 - {e}")
        plt.close(fig)
    else:
        print("  警告: Stage 2 數據不足，跳過繪圖。")

    # Plot Stage 3: Classification
    if metrics_data["s3_cls_train_loss"] and metrics_data["s3_cls_val_top1"]:
        print("  正在繪製 Stage 3 圖表...")
        epochs_s3 = range(1, len(metrics_data["s3_cls_train_loss"]) + 1)
        fig, ax1 = plt.subplots(figsize=(12, 6)) # 調整了figsize以匹配其他圖表，如果legend多可以再調大
        color = 'tab:red'
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Losses (Cls)', color=color)
        ax1.plot(epochs_s3, metrics_data["s3_cls_train_loss"], color=color, linestyle='-', label='S3 Cls Train Loss')
        if metrics_data["s3_seg_distill_loss"]:
            ax1.plot(epochs_s3, metrics_data["s3_seg_distill_loss"], color='tab:orange', linestyle=':', label='S3 Seg Distill')
        if metrics_data["s3_det_logits_distill_loss"]:
            ax1.plot(epochs_s3, metrics_data["s3_det_logits_distill_loss"], color='tab:green', linestyle='-.', label='S3 Det Logits Distill')
        if metrics_data["s3_det_feature_distill_loss"]:
             ax1.plot(epochs_s3, metrics_data["s3_det_feature_distill_loss"], color='tab:purple', linestyle='--', label='S3 Det Feature Distill')
        ax1.tick_params(axis='y', labelcolor=color)
        # 將圖例放在圖表下方，以避免與標題或Y軸重疊
        # ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2) 
        # 或者，如果線條不多，可以嘗試 'best' 或調整 loc
        ax1.legend(loc='upper left')


        ax2 = ax1.twinx(); color = 'tab:blue'
        ax2.set_ylabel('Validation Top-1 Acc (Cls)', color=color)
        ax2.plot(epochs_s3, metrics_data["s3_cls_val_top1"], color=color, linestyle='-', marker='o', label='S3 Val Top-1 Acc')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title('Stage 3: Classification Training Metrics') # 標題
        # fig.tight_layout(rect=[0, 0, 0.85, 0.95]) # 調整 rect 的 top 值，給標題留空間
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.90, hspace=0.25, wspace=0.25) # 更通用的調整


        try:
            save_path = os.path.join(output_dir, "stage3_classification_metrics.png")
            plt.savefig(save_path)
            print(f"  Stage 3 圖表已儲存到: {save_path}")
        except Exception as e:
            print(f"  錯誤: 儲存 Stage 3 圖表失敗 - {e}")
        plt.close(fig)
    else:
        print("  警告: Stage 3 數據不足，跳過繪圖。")
    print("所有圖表繪製完成。")

if __name__ == "__main__":
    log_file = "training_log.txt" 
    print(f"腳本將嘗試解析日誌檔案: {os.path.abspath(log_file)}")
    parsed_data = parse_log_file(log_file)
    if parsed_data:
        # 檢查是否有任何一個列表為空，這可能意味著解析不完整或日誌格式問題
        all_empty = True
        for key in parsed_data:
            if parsed_data[key]:
                all_empty = False
                break
        if all_empty:
            print("警告: 所有指標列表都為空，請檢查日誌檔案內容和正則表達式。")
        else:
            plot_metrics(parsed_data)
    else:
        print("日誌解析失敗，無法繪製圖表。")