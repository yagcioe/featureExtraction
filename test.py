import enviroment as env
import stft
import os
import random
import librosa
import numpy as np

env.override()

def loadData():
    dirPath = '/workspace/training/KEC/'
    index = os.listdir(dirPath)

    filePath = dirPath+random.choice(index)
    wav ,sr= librosa.load(filePath+'/speaker0.wav', mono=True,sr=env.sampleRate)
    print(wav)
    return wav

def transform():
    print(stft.transform(loadData()))

