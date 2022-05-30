
import math

"""Audio"""
sampleRate = 16000
"""STFT"""

iid = False
n_fft = 1024  # zero padding for stft
hop_Length = 256


"""LAbel generation"""
azimuth_slice_count = 37  # divide 180° into 36 slices with size of 5°
anlges_interval = [-math.pi, math.pi]

"""IO"""
source_dir = "/workspace/training/KEC/"
target_dir = "/workspace/featureData/"
split = [7,2,1] #train valid test , will get softmaxed