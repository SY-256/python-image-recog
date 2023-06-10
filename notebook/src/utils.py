from collections import deque
import copy
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from typing import Callable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

# t-SNEを使った特徴量のプロット関数
"""
t-SNEのプロット関数
data_loader : プロット対象のデータを読み込むデータローダ
model       : 特徴量抽出に使うモデル
num_samples : t-SNEでプロットするサンプル数
"""
def plot_t_sne(data_loader: DataLoader, model: nn.Module,
               num_samples: int):
    model.eval()
    
    # t-SNEのためのデータ整形
    x = []
    y = []
    for imgs, labels in data_loader:
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            
            # 特徴量の抽出
            embeddings = model(imgs, return_embed=True)
            
            x.append(embeddings.to('cpu'))
            y.append(labels.clone())
            
    x = torch.cat(x)
    y = torch.cat(y)
    
    # NumPy配列に変換
    x = x.numpy()
    y = y.numpy()
    
    # 指定サンプル数だけ抽出
    x = x[:num_samples]
    y = y[:num_samples]
    
    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=42)
    x_reduced = t_sne.fit_transform(x)
    
    # 各ラベルとマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']
    
    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
        plt.scatter(x_reduced[y==i, 0], x_reduced[y==i, 1],
                    c=[cmap(i/len(data_loader.dataset.classes))],
                    marker=markers[i], s=500, alpha=0.6, label=cls)
        plt.axis('off')
        plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
    plt.show()
    
# 評価関数
def evaluate(data_loader: DataLoader, model: nn.Module, loss_func: Callable):
    model.eval()
    
    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())
            
            y_pred = model(x)
            
            losses.append(loss_func(y_pred, y, reduction='none'))
            preds.append(y_pred.argmax(dim=1) == y)
            
    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()
    
    return loss, accuracy

# 各次元のデータセット全体の平均と標準偏差を計算する関数
"""
dataset: 平均と標準偏差を計算する対象のPytorchのデータセット
"""
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        # 3072次元のベクトルを取得
        img_flat = dataset[i][0]
        data.append(img_flat)
    # 第0軸を追加して第0軸でデータを連結
    data = np.stack(data)
    
    # データ全体の平均と標準偏差を計算
    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)
    
    return channel_mean, channel_std

# データセットを2つに分割するインデックス集合を生成する関数
"""
dataset    : 分割対象のデータセット
ration     : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
"""
def generate_subset(
    dataset: Dataset, ration: float, random_seed: int=42):
    # サブセットの大きさを計算
    size = int(len(dataset) * ration)
    
    indices = list(range(len(dataset)))
    
    # 2つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)
    
    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]
    
    return indices1, indices2