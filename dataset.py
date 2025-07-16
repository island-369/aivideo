# dataset.py (变长序列版)

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence # <--- 【新增】导入pad_sequence
import decord
from PIL import Image

decord.bridge.set_bridge('torch')

class VideoDataset(Dataset):
    # --- 【修改】在__init__中增加 debug_subset_size 参数 ---
    def __init__(self, train_dir, label_path, clip_processor, mode='train', split_ratio=0.9, debug_subset_size=None):
        super().__init__()
        
        self.train_dir = train_dir
        self.label_path = label_path
        self.mode = mode
        self.clip_processor = clip_processor
        
        with open(self.label_path,'r',encoding='utf-8') as f:
            self.labels_data = json.load(f)
        
        video_ids_in_folder = {f.split('.')[0] for f in os.listdir(self.train_dir) if f.endswith('.mp4')}
        video_ids_in_json = set(self.labels_data.keys())
        self.available_videos = list(video_ids_in_folder.intersection(video_ids_in_json))
        
        if not self.available_videos:
            raise FileNotFoundError(f"在目录 {self.train_dir} 中没有找到任何与标签文件匹配的视频。")
        
        # --- 【新增】限制数据集大小的逻辑 ---
        if debug_subset_size is not None:
            print(f"--- 调试模式开启：数据集将被限制为 {debug_subset_size} 个样本。 ---")
            # 为保证安全，即使设置的值大于总数，也只取总数
            size_to_take = min(debug_subset_size, len(self.available_videos))
            # 在划分前，先对总的可用视频列表进行截断
            self.available_videos = self.available_videos[:size_to_take]
        
        # 对（可能已被截断的）视频列表进行划分
        np.random.seed(42)
        np.random.shuffle(self.available_videos)
        
        split_point = int(len(self.available_videos) * split_ratio)
        if self.mode == 'train':
            self.video_files = self.available_videos[:split_point]
            print(f"训练集模式：共 {len(self.available_videos)} 个可用视频，加载 {len(self.video_files)} 个用于训练。")
        elif self.mode == 'val':
            self.video_files = self.available_videos[split_point:]
            print(f"验证集模式：共 {len(self.available_videos)} 个可用视频，加载 {len(self.video_files)} 个用于验证。")
        
        # ... 后续的数据增强和 __len__, __getitem__ 方法保持不变 ...
        self.test_transform = self.clip_processor
        original_transforms = self.clip_processor.transforms
        self.train_transform = transforms.Compose([
            original_transforms[0], original_transforms[1],
            transforms.RandomHorizontalFlip(p=0.5),
            original_transforms[2], original_transforms[3],
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self,idx):
        video_id = self.video_files[idx]
        video_path = os.path.join(self.train_dir, f"{video_id}.mp4")
        
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            frames_tensor = vr.get_batch(range(total_frames))
        except Exception as e:
            print(f"错误：加载视频失败 {video_path}: {e}")
            return None

        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
        
        transform = self.train_transform if self.mode == 'train' else self.test_transform
        processed_frames = torch.stack([transform(frame) for frame in frames_pil])
        
        label = float(self.labels_data[video_id]['is_fake'])
        
        return processed_frames, torch.tensor(label)

# ---【新增】一个专门用于填充和生成mask的collate_fn ---
def collate_fn_padding(batch):
    # 过滤掉加载失败的None值
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None, None
    
    # 将视频帧序列和标签分开
    sequences, labels = zip(*batch)
    
    # 1. 填充序列 (Padding)
    # pad_sequence 会自动将序列填充到这个batch中最长序列的长度
    # batch_first=True 使输出形状为 (B, T_max, C, H, W)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # 2. 创建掩码 (Masking)
    # 我们需要一个形状为 (B, T_max) 的mask，其中padding的位置为True，有效帧为False
    lengths = torch.tensor([s.shape[0] for s in sequences]) # 获取每个序列的原始长度
    max_len = padded_sequences.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    # 3. 堆叠标签
    labels = torch.stack(labels)
    
    return padded_sequences, labels, mask