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
from dataset import VideoDataset, collate_fn_padding

def main():
    # --- 1. DeepSpeed参数解析 ---
    parser = argparse.ArgumentParser(description="DeepSpeed Video Classifier Training")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- 2. 从我们自己的JSON文件加载配置 ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    train_config = config['train_config']
    model_config = config['model_config']

    # device由deepspeed的local_rank决定
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"

    # 只在主进程（rank 0）进行文件操作和打印
    save_dir = os.path.dirname(train_config['model_save_path'])
    if args.local_rank <= 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"使用设备: {device}")
        print("训练配置加载成功:", json.dumps(train_config, indent=2))

    # --- 3. 初始化模型和数据 ---
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
    
    # 注意：在多卡训练中，DataLoader不需要设置shuffle=True，因为DistributedSampler会处理shuffle
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])

    # --- 4. DeepSpeed 初始化 ---
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=None
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

    # --- 5. 训练循环 ---
    for epoch in range(train_config['epochs']):
        # 设置sampler的epoch，保证多卡数据shuffle同步
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if args.local_rank <= 0:
            print(f"\n--- Epoch {epoch+1}/{train_config['epochs']} ---")

        # --- 训练部分 ---
        model_engine.train()
        # 每个进程只计算自己的部分
        total_loss, correct_predictions, total_samples = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"训练中 (Rank {args.local_rank})", disable=(args.local_rank > 0))
        for batch in progress_bar:
            if not batch or batch[0] is None: continue
            frames, labels, mask, _ = batch # 忽略 video_ids
            frames = frames.to(device=model_engine.device, dtype=model_dtype)
            labels = labels.to(model_engine.device)
            mask = mask.to(model_engine.device)
            
            outputs = model_engine(frames, mask)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            # --- 【补全】指标计算逻辑 ---
            total_loss += loss.item() * frames.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            if args.local_rank <= 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 【补全】分布式指标汇总 ---
        # 将每个进程的统计结果转换为tensor
        total_loss_tensor = torch.tensor(total_loss, device=device)
        correct_predictions_tensor = torch.tensor(correct_predictions, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        
        # 使用all_reduce进行求和，结果会保存在每个进程的tensor中
        torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(correct_predictions_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        if args.local_rank <= 0:
            avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
            accuracy = correct_predictions_tensor.item() / total_samples_tensor.item()
            print(f"Epoch {epoch+1} 训练完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.4f}")

        # --- 验证部分 ---
        model_engine.eval()
        val_loss, val_corrects, val_samples = 0, 0, 0
        val_progress_bar = tqdm(val_loader, desc=f"验证中 (Rank {args.local_rank})", disable=(args.local_rank > 0))
        with torch.no_grad():
            for batch in val_progress_bar:
                if not batch or batch[0] is None: continue
                frames, labels, mask, _ = batch # 忽略 video_ids
                frames = frames.to(device=model_engine.device, dtype=model_dtype)
                labels = labels.to(model_engine.device)
                mask = mask.to(model_engine.device)
                
                outputs = model_engine(frames, mask)
                loss = criterion(outputs, labels)
                
                # --- 【补全】指标计算逻辑 ---
                val_loss += loss.item() * frames.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_corrects += (preds == labels).sum().item()
                val_samples += labels.size(0)
        
        # --- 【补全】分布式指标汇总 ---
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_corrects_tensor = torch.tensor(val_corrects, device=device)
        val_samples_tensor = torch.tensor(val_samples, device=device)

        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_corrects_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_samples_tensor, op=torch.distributed.ReduceOp.SUM)

        if args.local_rank <= 0:
            avg_val_loss = val_loss_tensor.item() / val_samples_tensor.item()
            val_accuracy = val_corrects_tensor.item() / val_samples_tensor.item()
            print(f"Epoch {epoch+1} 验证完成 | 平均损失: {avg_val_loss:.4f} | 准确率: {val_accuracy:.4f}")

            # --- 【补全】模型保存逻辑 ---
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_engine.save_checkpoint(save_dir, tag=os.path.basename(train_config['model_save_path']))
                print(f"发现新的最佳模型！准确率: {best_val_accuracy:.4f}，模型检查点已保存至 {save_dir}")

        # 添加同步屏障
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if args.local_rank <= 0:
        print("\n--- 训练结束 ---")
        print(f"最佳验证准确率: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()