import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import decord
import numpy as np
from collections import OrderedDict
import argparse
import deepspeed

# --- 【新增】为分布式推理导入必要的模块 ---
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import VideoClassifier

decord.bridge.set_bridge('torch')

CHUNK_SIZE = 500

# --- 【新增】一个专门用于推理的、更简洁的Dataset ---
class InferenceDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.video_files = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
    
    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.test_dir, video_file)
        return video_path, video_id

def predict(model_config, predict_config, args):
    # --- 【修改】device由deepspeed的local_rank决定 ---
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    if args.local_rank <= 0:
        print(f"使用设备: {device}")

    test_dir = predict_config['test_dir']
    model_path = predict_config['model_path']
    output_path = predict_config['output_path']

    # --- 加载模型 (每个进程都需要加载) ---
    model = VideoClassifier(model_config).half()
    if args.local_rank <= 0:
        print(f"正在从 {model_path} 加载模型权重...")
    state_dict = torch.load(model_path, map_location='cpu') # 先加载到CPU
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    clip_processor = model.clip_processor
    model_dtype = next(model.parameters()).dtype
    if args.local_rank <= 0:
        print(f"模型精度: {model_dtype}")

    # --- 【修改】使用分布式数据加载 ---
    dataset = InferenceDataset(test_dir)
    # DistributedSampler会自动为每个进程分配不重复的数据子集
    sampler = DistributedSampler(dataset, num_replicas=deepspeed.comm.get_world_size(), rank=args.local_rank, shuffle=False)
    # batch_size设为1，因为我们是逐个视频处理的
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=predict_config.get("num_workers", 2))

    local_results = [] # 每个进程只保存自己的结果

    with torch.no_grad():
        # 每个进程只处理自己的那一部分数据
        progress_bar = tqdm(dataloader, desc=f"正在预测 (Rank {args.local_rank})", disable=(args.local_rank > 0))
        for video_path, video_id in progress_bar:
            video_path, video_id = video_path[0], video_id[0] # 从batch中取出
            
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                is_fake_pred = False

                # --- 分块处理逻辑 (不变) ---
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
                    is_fake_pred = torch.sigmoid(output).item() > 0.5

                # --- 占位符生成逻辑 (不变) ---
                detail_placeholder = []
                if is_fake_pred:
                    try:
                        frame_data = vr[0]
                        height, width = frame_data.shape[0], frame_data.shape[1]
                        for i in range(total_frames):
                            detail_placeholder.append({"frame_idx": i, "bbox": [[0, 0, width, height]]})
                    except Exception as frame_e:
                        print(f"\n警告：无法为视频 {video_id} 解码帧以获取尺寸。错误: {frame_e}")
                
                # 将结果存入本地列表
                local_results.append({
                    "video_id": video_id,
                    "is_fake": is_fake_pred,
                    "detail": detail_placeholder
                })
                
            except Exception as e:
                print(f"\n处理视频 {video_path} 时出错: {e}")
                local_results.append({ "video_id": video_id, "is_fake": False, "detail": [] })

    # --- 【新增】结果汇总 ---
    # 等待所有进程都完成自己的推理任务
    deepspeed.comm.barrier()

    # 创建一个列表来接收所有进程的结果
    world_size = deepspeed.comm.get_world_size()
    all_results_list = [None] * world_size
    
    # 将 local_results 从当前进程发送到所有其他进程
    deepspeed.comm.all_gather_object(all_results_list, local_results)

    # --- 【新增】只在主进程（rank 0）合并和保存最终结果 ---
    if args.local_rank <= 0:
        print("\n所有进程已完成，正在合并结果...")
        final_results = {}
        # 遍历从所有进程收集到的结果列表
        for rank_results in all_results_list:
            for item in rank_results:
                final_results[item['video_id']] = {
                    "is_fake": item['is_fake'],
                    "detail": item['detail']
                }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
            
        print(f"预测完成，共 {len(final_results)} 条结果已保存至 {output_path}")

if __name__ == '__main__':
    # --- 【修改】使用argparse来接收deepspeed参数 ---
    parser = argparse.ArgumentParser(description="DeepSpeed Video Classifier Inference")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    args = parser.parse_args()
    
    # 初始化分布式环境
    deepspeed.init_distributed()

    print(f"开始执行预测脚本 (Rank {args.local_rank})...")
    with open('config.json', 'r') as f:
        main_config = json.load(f)
        
    predict(main_config['model_config'], main_config['predict_config'], args)