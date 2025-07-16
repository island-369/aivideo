# predict.py

import torch
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import decord
import numpy as np

from model import VideoClassifier # 导入我们训练的分类模型

decord.bridge.set_bridge('torch')

def predict(config, test_dir, model_path, output_path):
    """
    对测试集视频进行预测并生成提交文件
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 1. 加载模型 ---
    model_config = config['model_config']
    model = VideoClassifier(model_config)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    clip_processor = model.clip_processor

    # --- 2. 准备测试视频列表 ---
    test_videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
    print(f"找到 {len(test_videos)} 个测试视频。")
    
    results = {}

    # --- 3. 逐个视频进行预测 ---
    with torch.no_grad():
        for video_file in tqdm(test_videos, desc="正在预测"):
            video_id = video_file.split('.')[0]
            video_path = os.path.join(test_dir, video_file)
            
            try:
                # --- 【修改】核心逻辑：加载全部帧，与训练时保持一致 ---
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                frames_tensor = vr.get_batch(range(total_frames)) # 获取所有帧
                frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
                
                # b. 预处理 (与验证集逻辑完全相同)
                processed_frames = torch.stack([clip_processor(frame) for frame in frames_pil])
                processed_frames = processed_frames.unsqueeze(0).to(device) # 添加batch维度

                # c. 模型预测
                # 因为一次只处理一个视频，所以序列没有padding，mask全为False
                num_frames_in_video = processed_frames.shape[1]
                mask = torch.zeros(1, num_frames_in_video, dtype=torch.bool).to(device)
                
                output = model(processed_frames, mask)
                is_fake_prob = torch.sigmoid(output).item()
                
                # d. 生成预测结果
                is_fake_pred = is_fake_prob > 0.5
                
                # --- 定位信息的占位符 (逻辑不变) ---
                detail_placeholder = []
                if is_fake_pred:
                    # 您可以在这里添加一个默认策略，比如如果判断为假，就标记所有帧
                    # for i in range(total_frames):
                    #     detail_placeholder.append({
                    #         "frame_idx": i,
                    #         "bbox": [[0, 0, vr.width, vr.height]]
                    #     })
                    pass

                results[video_id] = {
                    "is_fake": is_fake_pred,
                    "detail": detail_placeholder
                }
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                results[video_id] = { "is_fake": False, "detail": [] }

    # --- 4. 保存结果 ---
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"预测完成，结果已保存至 {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="视频伪造检测推理脚本")
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试视频文件夹路径')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--output_path', type=str, default='submission.json', help='输出结果文件路径')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        main_config = json.load(f)
        
    predict(main_config, args.test_dir, args.model_path, args.output_path)