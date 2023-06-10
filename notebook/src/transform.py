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

