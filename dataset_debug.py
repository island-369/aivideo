import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import decord
from PIL import Image

decord.bridge.set_bridge('torch')

#输出每个进程处理的视频，确认是否重复处理

class VideoDataset(Dataset):
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
        
        if debug_subset_size is not None:
            print(f"--- 调试模式开启：数据集将被限制为 {debug_subset_size} 个样本。 ---")
            size_to_take = min(debug_subset_size, len(self.available_videos))
            self.available_videos = self.available_videos[:size_to_take]
        
        np.random.seed(42)
        np.random.shuffle(self.available_videos)
        
        split_point = int(len(self.available_videos) * split_ratio)
        if self.mode == 'train':
            self.video_files = self.available_videos[:split_point]
            # 为了避免在多卡时重复打印，可以将打印移到train.py的主进程中
            # print(f"训练集模式：共 {len(self.available_videos)} 个可用视频，加载 {len(self.video_files)} 个用于训练。")
        elif self.mode == 'val':
            self.video_files = self.available_videos[split_point:]
            # print(f"验证集模式：共 {len(self.available_videos)} 个可用视频，加载 {len(self.video_files)} 个用于验证。")
        
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
        
        # --- 【核心修改 1】: 返回值中增加 video_id ---
        return processed_frames, torch.tensor(label), video_id

def collate_fn_padding(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None, None, None # 返回四个None以匹配解包
    
    # --- 【核心修改 2】: 解包时增加 video_ids ---
    sequences, labels, video_ids = zip(*batch)
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([s.shape[0] for s in sequences])
    max_len = padded_sequences.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    labels = torch.stack(labels)
    
    # --- 【核心修改 3】: 返回值中增加 video_ids ---
    # video_ids 是一个元组(tuple) e.g., ('id1', 'id2', ...)
    return padded_sequences, labels, mask, video_ids