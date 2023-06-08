import torch
import librosa as librosa
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from scipy import signal


sfmax = nn.Softmax(dim=1)
# function extraction stft
def extract_stft(audio, sampling_rate=16000):

    fx, tx, stft_out = signal.stft(audio, sampling_rate, window="hann", nperseg=512, noverlap=256, nfft=512)
    stft_out = stft_out[:256, :]
    feat = np.concatenate(
        (np.abs(stft_out).reshape([stft_out.shape[0], stft_out.shape[1], 1]), np.angle(stft_out).reshape([stft_out.shape[0], stft_out.shape[1], 1])), axis=2
    )
    return feat


# reading audio clips
def audio_loading(path, sampling_rate=16000):

    audio, fs = librosa.load(path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    if fs != sampling_rate:
        audio = librosa.resample(audio, fs, sampling_rate)

    return audio


# function checking if the size of the inputs are same. If not, then the reference audio's size is adjusted
def check_size(audio_ref, audio_test):

    if len(audio_ref) > len(audio_test):
        print("Durations dont match. Adjusting duration of reference.")
        audio_ref = audio_ref[: len(audio_test)]

    elif len(audio_ref) < len(audio_test):
        print("Durations dont match. Adjusting duration of reference.")
        while len(audio_test) > len(audio_ref):
            audio_ref = np.append(audio_ref, audio_ref)
        audio_ref = audio_ref[: len(audio_test)]

    return audio_ref, audio_test


# audio loading and feature extraction
def feats_loading(test_path, ref_path=None, noresqa_or_noresqaMOS=0):

    if noresqa_or_noresqaMOS == 0 or noresqa_or_noresqaMOS == 1:

        audio_ref = audio_loading(ref_path)
        audio_test = audio_loading(test_path)
        audio_ref, audio_test = check_size(audio_ref, audio_test)

        if noresqa_or_noresqaMOS == 0:
            ref_feat = extract_stft(audio_ref)
            test_feat = extract_stft(audio_test)
            return ref_feat, test_feat
        else:
            return audio_ref, audio_test
