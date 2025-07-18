import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import json
import os
import argparse
import deepspeed

from model import VideoClassifier
from dataset_debug import VideoDataset, collate_fn_padding

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Video Classifier Training")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    train_config = config['train_config']
    model_config = config['model_config']
    
    train_config['device'] = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    
    if args.local_rank <= 0:
        save_dir = os.path.dirname(train_config['model_save_path'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"使用设备: {train_config['device']}")
        print("训练配置加载成功:", json.dumps(train_config, indent=2))
    
    model = VideoClassifier(model_config)
    clip_processor = model.clip_processor
    debug_size = train_config.get("debug_subset_size", None)

    train_dataset = VideoDataset(
        train_dir=train_config['data_dir'],
        label_path=train_config['label_path'],
        clip_processor=clip_processor,
        mode='train',
        debug_subset_size=debug_size
    )
    val_dataset = VideoDataset(
        train_dir=train_config['data_dir'],
        label_path=train_config['label_path'],
        clip_processor=clip_processor,
        mode='val',
        debug_subset_size=debug_size
    )

    if args.local_rank <= 0:
        print(f"训练集总数: {len(train_dataset)}")
        print(f"验证集总数: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), optimizer=None
    )

    criterion = nn.BCEWithLogitsLoss()
    best_val_accuracy = 0.0

    if model_engine.bfloat16_enabled():
        model_dtype = torch.bfloat16
        if args.local_rank <= 0: print("BF16 模式已启用。")
    elif model_engine.fp16_enabled():
        model_dtype = torch.float16
        if args.local_rank <= 0: print("FP16 模式已启用。")
    else:
        model_dtype = torch.float32
        if args.local_rank <= 0: print("FP32 模式已启用。")

    for epoch in range(train_config['epochs']):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        if args.local_rank <= 0:
            print(f"\n--- Epoch {epoch+1}/{train_config['epochs']} ---")
        
        # --- 训练部分 ---
        model_engine.train()
        
        # --- 【核心修改 1】: 在每个epoch开始时，为每个进程创建或清空日志文件 ---
        log_file_path = f"processed_videos_rank_{args.local_rank}.txt"
        with open(log_file_path, 'w' if epoch == 0 else 'a') as f:
            f.write(f"--- Epoch {epoch+1} ---\n")
        
        progress_bar = tqdm(train_loader, desc=f"训练中 (Rank {args.local_rank})", disable=(args.local_rank > 0))
        for batch in progress_bar:
            if not batch or batch[0] is None: continue
            
            # --- 【核心修改 2】: 解包时增加 video_ids ---
            frames, labels, mask, video_ids = batch

            # --- 【核心修改 3】: 将这个batch处理的video_id写入文件 ---
            with open(log_file_path, 'a') as f:
                for video_id in video_ids:
                    f.write(f"{video_id}\n")

            frames = frames.to(device=model_engine.device, dtype=model_dtype)
            labels = labels.to(model_engine.device)
            mask = mask.to(model_engine.device)
            
            outputs = model_engine(frames, mask)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

        # ... (后续验证和保存逻辑不变) ...

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if args.local_rank <= 0:
        print("\n--- 训练结束 ---")
        print(f"最佳验证准确率: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()