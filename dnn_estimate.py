import numpy as np
import yaml

import torch as t
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

from datasetloader import TestDataset
from net import Conv2dModel

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_score(score):
    return int(sigmoid(score)*100)

def dnn_estimate(sig, mode):
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    # 種目ごとに分類数が異なるので
    if mode == str(0):
        out_dim = 4
        model_src = './model/shoulders/shoulders.pth'
        condition_path = './model/shoulders/config.yaml'
    elif mode == str(1):
        out_dim = 4
        model_src = './model/abs_/abs.pth'
        condition_path = './model/abs_/config.yaml'
    elif mode == str(2):
        out_dim = 6
        model_src = './model/thighs_/thighs.pth'
        condition_path = './model/thighs_/config.yaml'

    with open(condition_path, 'r') as f:
        yml = yaml.safe_load(f)
    fft_size = yml['fft_size']['value']

    # DNNモデル定義
    if mode == str(1) or mode == str(2):
        model = Conv2dModel(fft_size, out_dim, mode='thighs').to(device)
    elif mode == str(0):
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_dim)

        model.fc = nn.Sequential(nn.Dropout(p=0), model.fc)
        model.layer4[1] = nn.Sequential(nn.Dropout(p=0), model.layer4[1])
        model.layer4[2] = nn.Sequential(nn.Dropout(p=0), model.layer4[2])

    # 保存されたモデルを読み込み
    model.load_state_dict(t.load(model_src, map_location=device))
    model.eval()

    dataset = TestDataset(sig, fft_size)
    dataloader = DataLoader(dataset, batch_size=1)

    for spec in dataloader:
        with t.no_grad():
            pred = model(spec)
    
    _, category = t.max(pred.data, 1)

    softmax = nn.Softmax(dim=1)
    scores = softmax(pred)

    # 腕と腹筋
    if mode == str(0) or mode == str(1):
        score = scores[0,0]
    # ももあげ
    elif mode == str(2):
        score = scores[0,-1]
    
    score = score.cpu().detach().numpy().item()
    category = category.cpu().detach().numpy().item()

    return category, calc_score(score)

if __name__ == '__main__':

    sig = np.loadtxt('./2_4.csv', delimiter=',')
    sig = sig.T

    cl, score = dnn_estimate(sig, mode='2')
    print(cl, score)