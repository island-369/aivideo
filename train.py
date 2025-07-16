# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os # 导入os

from model import VideoClassifier
from dataset import VideoDataset, collate_fn_padding



def main():
    # --- 1. 从JSON文件加载配置 ---
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    train_config = config['train_config']
    model_config = config['model_config']
    
    train_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 【新增】确保模型保存目录存在
    save_dir = os.path.dirname(train_config['model_save_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"使用设备: {train_config['device']}")
    print("配置加载成功:", json.dumps(config, indent=2))
    
    # --- 2. 初始化模型 ---
    model = VideoClassifier(model_config) 
    clip_processor = model.clip_processor
    model.to(train_config['device'])
    
    # print(f"模型加载完成:{model}")
    # --- 3. 准备数据 ---
    # 【修改】初始化Dataset时不再需要num_frames
    train_dataset = VideoDataset(
        train_dir=train_config['data_dir'], 
        label_path=train_config['label_path'],
        clip_processor=clip_processor,
        mode='train'
    )
    val_dataset = VideoDataset(
        train_dir=train_config['data_dir'], 
        label_path=train_config['label_path'],
        clip_processor=clip_processor,
        mode='val'
    )
    
    # 【修改】为DataLoader使用新的collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        collate_fn=collate_fn_padding, # <--- 使用新的collate_fn
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'] * 2,
        shuffle=False,
        num_workers=train_config['num_workers'],
        collate_fn=collate_fn_padding, # <--- 使用新的collate_fn
        pin_memory=True
    )
    
    # --- 4. 定义损失函数和优化器 ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    
    # --- 【新增】逻辑：用于保存最佳模型 ---
    best_val_accuracy = 0.0

    # --- 5. 训练循环 ---
    for epoch in range(train_config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{train_config['epochs']} ---")
        
        # --- 训练部分 ---
        model.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        progress_bar = tqdm(train_loader, desc="训练中")
        for batch in progress_bar:
            if not batch or batch[0] is None: continue
            
            frames, labels, mask = batch
            frames, labels, mask = frames.to(train_config['device']), labels.to(train_config['device']), mask.to(train_config['device'])
            
            optimizer.zero_grad()
            outputs = model(frames, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds == labels).sum().item()/labels.size(0):.2f}")

        if total_samples > 0:
            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1} 训练完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.4f}")

        # --- 【新增】验证部分 ---
        model.eval()
        val_loss, val_corrects, val_samples = 0, 0, 0
        val_progress_bar = tqdm(val_loader, desc="验证中")
        with torch.no_grad():
            for batch in val_progress_bar:
                if not batch or batch[0] is None: continue

                frames, labels, mask = batch
                frames, labels, mask = frames.to(train_config['device']), labels.to(train_config['device']), mask.to(train_config['device'])
                
                outputs = model(frames, mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_corrects += (preds == labels).sum().item()
                val_samples += labels.size(0)

        if val_samples > 0:
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_corrects / val_samples
            print(f"Epoch {epoch+1} 验证完成 | 平均损失: {avg_val_loss:.4f} | 准确率: {val_accuracy:.4f}")

            # --- 【新增】模型检查点逻辑 ---
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), train_config['model_save_path'])
                print(f"发现新的最佳模型！准确率: {best_val_accuracy:.4f}，模型已保存至 {train_config['model_save_path']}")

    print("\n--- 训练结束 ---")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()