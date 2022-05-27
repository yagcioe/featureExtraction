
import math

"""Audio"""
sampleRate = 16000
"""STFT"""

iid = False
n_fft = 1024  # zero padding for stft
hop_Length = 256


"""LAbel generation"""
azimuth_slice_count = 9  # divide 180° into 36 slices with size of 5°
anlges_interval = [-math.pi, math.pi]

"""IO"""
source_dir = "worspace/training/KEC/"
target_dir = "workspace/featureData/"
split = [70,20,10] #train valid test , will get softmaxed