import numpy as np
from torch.utils.data import Dataset
from scipy.fftpack import fft
from numpy import hamming

epsilon = np.finfo(float).eps

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
    
    # 振幅スペクトログラム
    S = np.abs(S)

    # フロアリング
    S = np.where(S < epsilon, S+epsilon, S)
    S = np.log(S)

    return S

class TestDataset(Dataset):

    def __init__(self, sig, fft_size=128):

        self.sig = sig
        self.fft_size = fft_size

    def __len__(self):
        return 1
    
    def standard_scale(self, sig):
        std = np.std(sig, axis=0)
        mean = np.mean(sig, axis=0)
        return (sig - mean) / (std + (epsilon*np.random.rand()))

    def __getitem__(self, idx):

        sig_norm = self.standard_scale(self.sig)

        spec_list = []

        for i in range(self.sig.shape[1]):
            spec = sig2spec(sig_norm[:, i], self.fft_size, self.fft_size//2)
            spec_list.append(spec)

        spec_ = np.array(spec_list, dtype=np.float32)

        return spec_