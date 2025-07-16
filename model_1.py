import torch
import torch.nn as nn
import open_clip
import math



class VideoClassifier(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        
        clip_model_name=model_config['clip_model_name']
        pretrained=model_config['pretrained']
        num_transformer_layers=model_config['num_transformer_layers']
        num_heads=model_config['num_heads']
        hidden_dim=model_config['hidden_dim']
        mlp_dim=model_config['mlp_dim']
        dropout=model_config['dropout']
        
        
        print("正在加载clip模型")
        self.clip_model,_,self.clip_processor=open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained
        )
        
        
        for param in self.clip_model.parameters():
            param.requires_grad=False
            
        print("clip模型加载完成")  
        
        # #定义transformer编码器
        # self.clip_feature_dim=self.clip_model.visual.output_dim
        
        # #定义一个线性层将clip输出维度映射到transformer到隐藏维度
        # self.feature_projection=nn.Linear(self.clip_feature_dim,hidden_dim)
        
        # encoder_layer=nn.TransformerEncoderLayer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=hidden_dim*4,
        #     dropout=dropout,
        #     batch_first=True
        # )
        
        # self.positional_encoding=nn.Parameter(torch.zeros())
        
    # def forward(self,x,mask):