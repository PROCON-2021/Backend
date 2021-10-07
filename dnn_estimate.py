import numpy as np
import torch as t
import torch.nn as nn
from net import Conv2dModel
from torchvision import models

import numpy as np
from scipy.fftpack import fft
from numpy import hamming
import yaml

def sig2spec(src, fft_size, shift_size):

    signal = np.array(src)
    window = hamming(fft_size+1)[:fft_size]

    zeroPadSize = fft_size - shift_size
    length = signal.shape[0]
    frames = int(np.floor((length + fft_size - 1) / shift_size))
    I = int(fft_size/2 + 1)

    signal = np.concatenate([np.zeros(zeroPadSize), signal, np.zeros(fft_size)])
    S = np.zeros([I, frames], dtype=np.complex128)

    for j in range(frames):
        sp = j * shift_size
        spectrum = fft(signal[sp: sp+fft_size] * window)
        S[:, j] = spectrum[:I]

    return S

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_score(score):
    return int(sigmoid(score)*100)

def standard_scale(sig):
    std = np.std(sig, axis=0)
    mean = np.mean(sig, axis=0)
    return (sig - mean) / (std+np.finfo(float).eps)

def dnn_estimate(sig, mode):
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    # 種目ごとに分類数が異なるので
    if mode == str(0):
        out_dim = 4
        model_src = './model/shoulders.pth'
        condition_path = './model/shoulders/config.yaml'
    elif mode == str(1):
        out_dim = 4
        model_src = './model/abs/abs.pth'
        condition_path = './model/abs/config.yaml'
    elif mode == str(2):
        out_dim = 6
        model_src = './model/thighs.pth'
        condition_path = './model/thighs/config.yaml'

    with open(condition_path, 'r') as f:
        yml = yaml.safe_load(f)
    fft_size = yml['fft_size']['value']

    # DNNモデル定義
    # model = Conv2dModel(out_dim).to(device)
    model = models.resnet152(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_dim)

    model.fc = nn.Sequential(nn.Dropout(p=0), model.fc)
    model.layer4[1] = nn.Sequential(nn.Dropout(p=0), model.layer4[1])
    model.layer4[2] = nn.Sequential(nn.Dropout(p=0), model.layer4[2])

    # 保存されたモデルを読み込み
    model.load_state_dict(t.load(model_src, map_location=device))
    model.eval()

    # 直流成分を差し引いて正規化

    sig_ = standard_scale(sig)

    spec_list = []

    for i in range(sig.shape[1]):
        spec_ = np.abs(sig2spec(sig_[:, i], fft_size, fft_size//2)).astype('float32')
        spec_list.append(spec_)
    
    spec = np.array(spec_list)

    spec = t.from_numpy(spec[np.newaxis,:,:,:]).to(device)

    with t.no_grad():
        pred = model(spec)
    
    _, category = t.max(pred.data, 1)

    softmax = nn.Softmax(dim=1)
    scores = softmax(pred)
    score = scores[0,0]

    score = score.cpu().detach().numpy()
    category = category.cpu().detach().numpy()

    return category.item(), calc_score(score.item())