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

# 画像整形関数
"""
img         : 整形対象の画像
channel_mean: 各次元のデータセット全体の平均、[入力次元]
channel_std : 各次元のデータセット全体の標準偏差、[入力次元]  
"""
def transform(img: Image.Image, channel_mean: np.ndarray=None,
              channel_std: np.ndarray=None):
    # PILからNumPy配列に変換
    img = np.asarray(img, dtype='float32')
    
    # [32, 32, 3]の画像を3072次元のベクトルに平坦化
    x = img.flatten()
    
    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
    
    return x

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