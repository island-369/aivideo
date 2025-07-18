import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import decord
import numpy as np
from collections import OrderedDict

from model import VideoClassifier # 导入我们训练的分类模型

decord.bridge.set_bridge('torch')

def predict(model_config, predict_config):
    """
    对测试集视频进行预测并生成提交文件
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 1. 从配置中获取路径 ---
    test_dir = predict_config['test_dir']
    model_path = predict_config['model_path']
    output_path = predict_config['output_path']

    # --- 2. 加载模型 ---
    model = VideoClassifier(model_config)
    
    print(f"正在从 {model_path} 加载模型权重...")
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    clip_processor = model.clip_processor
    model_dtype = next(model.parameters()).dtype
    print(f"模型精度: {model_dtype}")

    # --- 3. 准备测试视频列表 ---
    test_videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
    print(f"找到 {len(test_videos)} 个测试视频。")
    
    results = {}

    # --- 4. 逐个视频进行预测 ---
    with torch.no_grad():
        for video_file in tqdm(test_videos, desc="正在预测"):
            video_id = video_file.split('.')[0]
            video_path = os.path.join(test_dir, video_file)
            
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                frames_tensor = vr.get_batch(range(total_frames))
                frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
                
                processed_frames = torch.stack([clip_processor(frame) for frame in frames_pil])
                processed_frames = processed_frames.unsqueeze(0).to(device=device, dtype=model_dtype)

                num_frames_in_video = processed_frames.shape[1]
                mask = torch.zeros(1, num_frames_in_video, dtype=torch.bool).to(device)
                
                output = model(processed_frames, mask)
                is_fake_prob = torch.sigmoid(output).item()
                is_fake_pred = is_fake_prob > 0.5
                
                # --- 【核心修改】 开始 ---
                detail_placeholder = []
                if is_fake_pred:
                    # 如果判断为伪造，则为所有帧生成一个覆盖全画面的bbox
                    # decord可以直接获取视频的宽和高
                    width = vr.width
                    height = vr.height
                    
                    for i in range(total_frames):
                        detail_placeholder.append({
                            "frame_idx": i,
                            "bbox": [[0, 0, width, height]]
                        })
                # --- 【核心修改】 结束 ---

                results[video_id] = {
                    "is_fake": is_fake_pred,
                    "detail": detail_placeholder
                }
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                results[video_id] = { "is_fake": False, "detail": [] }

    # --- 5. 保存结果 ---
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False) # indent=2 减小文件大小
        
    print(f"预测完成，结果已保存至 {output_path}")

if __name__ == '__main__':
    print("开始执行预测脚本...")
    with open('config.json', 'r') as f:
        main_config = json.load(f)
        
    predict(main_config['model_config'], main_config['predict_config'])