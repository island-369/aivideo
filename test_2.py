import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import decord
import numpy as np
from collections import OrderedDict

from model import VideoClassifier

decord.bridge.set_bridge('torch')

CHUNK_SIZE = 500

def predict(model_config, predict_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    test_dir = predict_config['test_dir']
    model_path = predict_config['model_path']
    output_path = predict_config['output_path']

    # --- 加载模型 ---
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

    test_videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
    print(f"找到 {len(test_videos)} 个测试视频。")
    
    results = {}

    with torch.no_grad():
        for video_file in tqdm(test_videos, desc="正在预测"):
            video_id = video_file.split('.')[0]
            video_path = os.path.join(test_dir, video_file)
            
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                is_fake_pred = False

                # --- 预测逻辑 ---
                if total_frames > CHUNK_SIZE:
                    chunk_probs = []
                    for i in range(0, total_frames, CHUNK_SIZE):
                        indices = range(i, min(i + CHUNK_SIZE, total_frames))
                        frames_tensor = vr.get_batch(indices)
                        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
                        
                        processed_frames = torch.stack([clip_processor(frame) for frame in frames_pil])
                        processed_frames = processed_frames.unsqueeze(0).to(device=device, dtype=model_dtype)

                        mask = torch.zeros(1, processed_frames.shape[1], dtype=torch.bool).to(device)
                        output = model(processed_frames, mask)
                        chunk_probs.append(torch.sigmoid(output).item())
                    
                    if any(p > 0.5 for p in chunk_probs):
                        is_fake_pred = True
                else:
                    frames_tensor = vr.get_batch(range(total_frames))
                    frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
                    
                    processed_frames = torch.stack([clip_processor(frame) for frame in frames_pil])
                    processed_frames = processed_frames.unsqueeze(0).to(device=device, dtype=model_dtype)

                    mask = torch.zeros(1, processed_frames.shape[1], dtype=torch.bool).to(device)
                    output = model(processed_frames, mask)
                    is_fake_prob = torch.sigmoid(output).item()
                    is_fake_pred = is_fake_prob > 0.5

                # --- 占位符生成逻辑 ---
                detail_placeholder = []
                if is_fake_pred:
                    # --- 【核心修改】 ---
                    # 只有在确定视频是伪造的情况下，我们才需要解码第一帧来获取尺寸，这样更高效
                    try:
                        frame_data = vr[0]
                        height, width = frame_data.shape[0], frame_data.shape[1]
                        
                        for i in range(total_frames):
                            detail_placeholder.append({
                                "frame_idx": i,
                                "bbox": [[0, 0, width, height]]
                            })
                    except Exception as frame_e:
                        print(f"\n警告：无法为视频 {video_id} 解码帧以获取尺寸。错误: {frame_e}")
                        # 即使获取尺寸失败，也保持 is_fake=True，但 detail 为空
                        pass
                # --- 【核心修改结束】 ---

                results[video_id] = {
                    "is_fake": is_fake_pred,
                    "detail": detail_placeholder
                }
                
            except Exception as e:
                print(f"\n处理视频 {video_path} 时发生严重错误: {e}")
                results[video_id] = { "is_fake": False, "detail": [] }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"预测完成，结果已保存至 {output_path}")

if __name__ == '__main__':
    print("开始执行预测脚本...")
    with open('config.json', 'r') as f:
        main_config = json.load(f)
        
    predict(main_config['model_config'], main_config['predict_config'])