import soundfile as sf
MAX_WAV_VALUE = 32768.0
import torch
import os
from scipy.io.wavfile import write
import numpy as np
import librosa


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


def torch_stft(input, nfft=2048, center=True):
    in_numpy = input.clone().detach()
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