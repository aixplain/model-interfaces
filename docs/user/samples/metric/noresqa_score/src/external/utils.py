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


# noresqa and noresqa-mos prediction calls
def model_prediction_noresqa(test_feat, nmr_feat, model):

    intervals_sdr = np.arange(0.5, 40, 1)

    with torch.no_grad():
        ranking_frame, sdr_frame, snr_frame = model(test_feat.permute(0, 3, 2, 1), nmr_feat.permute(0, 3, 2, 1))
        # preference task prediction
        ranking = sfmax(ranking_frame).mean(2).detach().cpu().numpy()
        pout = ranking[0][0]
        # quantification task
        sdr = intervals_sdr * (sfmax(sdr_frame).mean(2).detach().cpu().numpy())
        qout = sdr.sum()

    return pout, qout


def model_prediction_noresqa_mos(test_feat, nmr_feat, model):

    with torch.no_grad():
        score = model(nmr_feat, test_feat).detach().cpu().numpy()[0]

    return score


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
