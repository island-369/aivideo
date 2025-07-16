# model.py
import torch
import torch.nn as nn
import open_clip
import math

class SinusoidalPositionalEncoding(nn.Module):
    # ... (这部分代码是正确的，保持不变) ...
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VideoClassifier(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        hidden_dim = model_config['hidden_dim']
        dropout = model_config['dropout']
        
        print("正在加载 CLIP 模型...")
        self.clip_model, _, self.clip_processor = open_clip.create_model_and_transforms(
            model_config['clip_model_name'], pretrained=model_config['pretrained']
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("CLIP 模型加载完成。")

        # --- 【模型手术 开始】 ---
        # 1. 直接获取视觉主干网络
        self.visual_trunk = self.clip_model.visual.trunk
        
        # 2. 禁用其内部的池化和展平层
        self.visual_trunk.head.global_pool = nn.Identity()
        self.visual_trunk.head.flatten = nn.Identity()
        
        # 3. 创建我们自己的池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 从 visual.head.proj 读取输入维度，这是最可靠的方式
        self.clip_feature_dim = self.clip_model.visual.head.proj.in_features
        print(f"精确找到的CLIP特征维度: {self.clip_feature_dim}")
        
        self.feature_projection = nn.Linear(self.clip_feature_dim, hidden_dim)
        # --- 【模型手术 结束】 ---

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=model_config['num_heads'], 
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config['num_transformer_layers'])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_config['mlp_dim']),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_config['mlp_dim'], 1)
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_dim, 
            dropout=dropout, 
            max_len=512 
        )

    def forward(self, x, mask):
        batch_size, num_frames, c, h, w = x.shape
        
        x = x.view(batch_size * num_frames, c, h, w)
        with torch.no_grad():
            # --- 【模型手术对应的Forward流程】 ---
            # 1. 通过改造后的主干网络，获取特征图 (B*T, 3072, H', W')
            feature_map = self.visual_trunk(x)
            # 2. 使用我们自己的池化层和展平操作，得到特征向量 (B*T, 3072)
            frame_features = self.avgpool(feature_map).flatten(1)

        frame_features = frame_features.view(batch_size, num_frames, -1)

        # 后续流程完全不变，因为我们成功拿到了3072维的特征
        projected_features = self.feature_projection(frame_features)
        projected_features = self.positional_encoding(projected_features)
        transformer_output = self.transformer_encoder(
            src=projected_features, 
            src_key_padding_mask=mask
        )
        inverted_mask = ~mask
        masked_output = transformer_output * inverted_mask.unsqueeze(-1)
        sum_output = masked_output.sum(dim=1)
        valid_lengths = inverted_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        mean_output = sum_output / valid_lengths
        logits = self.mlp_head(mean_output)
        
        return logits.squeeze(-1)