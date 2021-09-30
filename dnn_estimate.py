import numpy as np
import torch as t
import torch.nn as nn
from net import Conv2dModel

import numpy as np
from scipy.fftpack import fft
from numpy import hamming

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

def calc_score(score):
    return int(score * 100)

def dnn_estimate(sig, mode):
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    # 種目ごとに分類数が異なるので
    if mode == str(0):
        out_dim = 4
        model_src = './model/shoulders.pth'
    if mode == str(1):
        out_dim = 4
        model_src = './model/abs.pth'
    elif mode == str(2):
        out_dim = 6
        model_src = './model/thighs.pth'

    # DNNモデル定義
    model = Conv2dModel(out_dim).to(device)

    # 保存されたモデルを読み込み
    model.load_state_dict(t.load(model_src, map_location=device))
    model.eval()

    # 直流成分を差し引いて正規化
    sig_ = (sig - 512) / 1023

    spec_list = []

    for i in range(sig.shape[1]):
        spec_ = np.abs(sig2spec(sig_[:, i], 128, 64)).astype('float32')
        spec_list.append(spec_)
    
    spec = np.array(spec_list)

    spec = t.from_numpy(spec[np.newaxis,:,:,:]).to(device)

    with t.no_grad():
        pred = model(spec)
    
    _, category = t.max(pred.data, 1)

    softmax = nn.Softmax(dim=1)
    scores = softmax(pred)
    score, _ = t.max(scores, 1)

    score = score.cpu().detach().numpy()
    category = category.cpu().detach().numpy()

    return category.item(), calc_score(score.item())