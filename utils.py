import soundfile as sf
import torch
import os
from scipy.io.wavfile import write
import numpy as np
import librosa
from scipy.special import expi
import librosa.display
import matplotlib.pyplot as plt
MAX_WAV_VALUE = 32768.0


class Accumulator:
    def __init__(self, noisy_audio, low_cut, high_cut, nfft, center, residual, sr, bandpass):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.center = center
        self.nfft = nfft
        self.stft_noisy = torch_stft(noisy_audio, nfft=nfft, center=center)
        self.residual = residual
        self.sr = sr
        self.bandpass = bandpass
        self.stft_noisy_filt = self.filter_stft()
        self.stft_diff_sum = np.zeros(self.stft_noisy.shape)

    def filter_stft(self):
        stft_full = self.stft_noisy + 0
        stft_full[:self.bandpass, :] = 0 * stft_full[:self.bandpass, :]  # reduce low frequencies
        stft_full[-self.bandpass // 3:, :] = 0 * stft_full[-self.bandpass // 3:, :]  # reduce high frequencies
        return stft_full

    def sum_difference(self, stft, iter_num):
        if iter_num < 48:
            self.stft_prev, self.stft_minus = stft, stft
        else:
            self.stft_minus = np.abs(stft - self.stft_prev)/(stft+np.finfo(float).eps)
            self.stft_diff_sum += self.stft_minus
            self.stft_diff_sum[self.stft_diff_sum < np.percentile(self.stft_diff_sum, self.low_cut)] = \
                np.percentile(self.stft_diff_sum, self.low_cut)
            self.stft_diff_sum[self.stft_diff_sum > np.percentile(self.stft_diff_sum, self.high_cut)] = \
                np.percentile(self.stft_diff_sum, self.high_cut)
            self.stft_prev = stft

    def create_atten_map(self):
        max_mask = self.stft_diff_sum.max()
        min_mask = self.stft_diff_sum.min()
        atten_map = (max_mask - self.stft_diff_sum) / (max_mask - min_mask)
        atten_map[atten_map < self.residual] = self.residual
        self.atten_map = atten_map

    def mmse_lsa(self):
        gamma_mat = (1 - self.atten_map) ** 2
        gamma_mat[gamma_mat < 10**-10] = 10**-10
        gamma_mat = 1 / gamma_mat
        lsa_mask = np.zeros(gamma_mat.shape)
        for it, gamma in enumerate(gamma_mat.transpose()):
            eta = gamma - 1
            eta[eta < self.residual] = self.residual
            v = gamma * eta / (1 + eta)
            gain = np.ones(gamma.shape)
            idx = v > 5
            gain[idx] = eta[idx] / (1 + eta[idx])
            idx = np.logical_and(v <= 5, v > 0)  # and v > 0
            gain[idx] = eta[idx] / (1 + eta[idx]) * np.exp(0.5 * -expi(-v[idx]))
            gain[gain > 1] = 1
            lsa_mask[:, it] = gain
        self.lsa_mask = lsa_mask

    def show_lsa(self):
        plot_stft(self.lsa_mask, 'LSA Mask')

    def show_wiener(self):
        plot_stft(self.lsa_mask, 'Wiener Mask')

    def show_diff_accum(self):
        plot_stft(self.stft_diff_sum, 'Accumulator')

    def show_diff_stft(self):
        plot_stft(self.stft_minus, 'Abs Difference')

    def show_noisy(self):
        plot_stft(np.abs(self.stft_noisy), 'Power Spectrogram of Noisy Sample')


def plot_stft(D, title=''):
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def write_norm_music(input, filename, sr):
    audio = MAX_WAV_VALUE * input
    wavdata = audio.astype('int16')
    write(filename, sr, wavdata)


def torch_stft(audio, nfft=2048, center=True):
    in_numpy = audio.clone().detach()
    in_numpy = in_numpy.detach().cpu().numpy()
    stft = librosa.stft(in_numpy, n_fft=nfft, center=center)
    return stft


def write_music_stft(stft, filename, sr, center=True):
    in_numpy = librosa.istft(stft, center=center)
    write_norm_music(in_numpy, filename, sr)


def makedirs(outputs_dir):
    if not os.path.exists(outputs_dir):
        print("Creating directory: {}".format(outputs_dir))
        os.makedirs(outputs_dir)