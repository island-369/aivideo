import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    # 添加deepspeed自己的参数，它会自动寻找ds_config.json
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- 2. 从我们自己的JSON文件加载配置 ---
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    train_config = config['train_config']
    model_config = config['model_config']
    
    # device由deepspeed的local_rank决定
    train_config['device'] = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    
    # 只在主进程（rank 0）进行文件操作和打印
    if args.local_rank <= 0:
        save_dir = os.path.dirname(train_config['model_save_path'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"使用设备: {train_config['device']}")
        print("训练配置加载成功:", json.dumps(train_config, indent=2))
    
    # --- 3. 初始化模型和数据 ---
    # 这部分逻辑不变
    model = VideoClassifier(model_config) 
    clip_processor = model.clip_processor
    
    debug_size = train_config.get("debug_subset_size", None)
    
    # --- 【修改】将debug_size传递给Dataset ---
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
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'] * 2, collate_fn=collate_fn_padding, num_workers=train_config['num_workers'])
    
    # --- 4. DeepSpeed 初始化 ---
    # 【核心修改】我们不再手动创建 optimizer 对象
    # optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate']) # <-- 删除或注释掉这一行
    
    # 【核心修改】将 optimizer 设为 None，并传入 model_parameters，让 DeepSpeed 自己创建最高效的优化器
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=None 
    )
    
    criterion = nn.BCEWithLogitsLoss()
    best_val_accuracy = 0.0

    # --- 5. 训练循环 ---
    for epoch in range(train_config['epochs']):
        # 只在主进程打印Epoch信息
        if args.local_rank <= 0:
            print(f"\n--- Epoch {epoch+1}/{train_config['epochs']} ---")
        
        # --- 训练部分 ---
        model_engine.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        
        # disable=True/False 控制只在主进程显示tqdm进度条
        progress_bar = tqdm(train_loader, desc="训练中", disable=(args.local_rank > 0))
        for batch in progress_bar:
            if not batch or batch[0] is None: continue
            
            frames, labels, mask = batch
            # frames, labels, mask = frames.to(model_engine.device), labels.to(model_engine.device), mask.to(model_engine.device)
            # train_deepspeed.py -> 训练循环内

            # 将frames张量在移动到设备的同时，转换数据类型为float16
            frames = frames.to(device=model_engine.device, dtype=torch.float16)
            labels = labels.to(model_engine.device)
            mask = mask.to(model_engine.device)
            
            # 使用 model_engine 进行前向、反向传播和更新
            outputs = model_engine(frames, mask)
            loss = criterion(outputs, labels)
            
            model_engine.backward(loss)
            model_engine.step()

            # --- 指标计算（可以在所有进程上算，但只在主进程打印） ---
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            if args.local_rank <= 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds == labels).sum().item()/labels.size(0):.2f}")

        if total_samples > 0 and args.local_rank <= 0:
            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1} 训练完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.4f}")

        # --- 验证部分 ---
        model_engine.eval()
        val_loss, val_corrects, val_samples = 0, 0, 0
        val_progress_bar = tqdm(val_loader, desc="验证中", disable=(args.local_rank > 0))
        with torch.no_grad():
            for batch in val_progress_bar:
                if not batch or batch[0] is None: continue
                frames, labels, mask = batch
                # frames, labels, mask = frames.to(model_engine.device), labels.to(model_engine.device), mask.to(model_engine.device)
                # 验证时同样需要转换数据类型
                frames = frames.to(device=model_engine.device, dtype=torch.float16)
                labels = labels.to(model_engine.device)
                mask = mask.to(model_engine.device)
                
                outputs = model_engine(frames, mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_corrects += (preds == labels).sum().item()
                val_samples += labels.size(0)

        if val_samples > 0 and args.local_rank <= 0:
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_corrects / val_samples
            print(f"Epoch {epoch+1} 验证完成 | 平均损失: {avg_val_loss:.4f} | 准确率: {val_accuracy:.4f}")

            # 只在主进程保存模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # 使用DeepSpeed的保存方法，它会处理好分布式状态
                model_engine.save_checkpoint(os.path.dirname(train_config['model_save_path']), tag=os.path.basename(train_config['model_save_path']))
                print(f"发现新的最佳模型！准确率: {best_val_accuracy:.4f}，模型检查点已保存至 {os.path.dirname(train_config['model_save_path'])}")

    if args.local_rank <= 0:
        print("\n--- 训练结束 ---")
        print(f"最佳验证准确率: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()