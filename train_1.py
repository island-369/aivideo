import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model_1 import VideoClassifier
# from dataset import VideoDataset, collate_fn


def main():
    with open('config.json','r',encoding='utf-8') as f:
        config=json.load(f)
    
    train_config=config['train_config']
    model_config=config['model_config']

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")
    train_config['device']=str(device)
    print(f"训练配置加载:{json.dumps(train_config,indent=2)}")

    #初始化模型
    model=VideoClassifier(model_config)
    clip_processor=model.clip_processor
    model.to(train_config['device'])
    print(f"模型加载完成:{model}")
    
    
    
if __name__=='__main__':
    main()