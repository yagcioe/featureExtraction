import math
import os
import librosa
from . import enviroment as env
import numpy as np


def transform(wav):  # Freq Bins x Time Bins
    return librosa.stft(wav, n_fft=env.n_fft, hop_length=env.hop_Length)


"""Utils"""



def to_phase(D):
    return np.angle(D)

def to_ipd(D,E):
    diff = to_phase(D)-to_phase(E)
    return boundAngle(diff)



def to_intensity(D):
   return np.abs(D)

def to_dB(D):
    return librosa.amplitude_to_db(D,ref=0)


def freqBins():
    return librosa.fft_frequencies(sr=env.sampleRate, n_fft=env.n_fft)


def frame_to_sample(frame):
    return librosa.frames_to_samples(frame, hop_length=env.hop_Length, n_fft=env.n_fft)


def frame_to_time(frame):
    return librosa.frames_to_time(frame, sr=env.sampleRate, hop_length=env.hop_Length, n_fft=env.n_fft)


def samples_to_frame(samples):
    return librosa.samples_to_frames(samples, hop_length=env.hop_Length, n_fft=env.n_fft)


def samples_to_time(sample):
    return librosa.samples_to_time(sample, env.sampleRate)


def time_to_samples(time):
    return librosa.time_to_samples(time, env.sampleRate)


def time_to_frame(time):
    return librosa.time_to_frames(time, sr=env.sampleRate, hop_length=env.hop_Length, n_fft=env.n_fft)

def boundAngle(phi):
    return phi % (2*math.pi)

def ensurePath(path):
    if(not os.path.isdir(path)):
        os.makedirs(path)