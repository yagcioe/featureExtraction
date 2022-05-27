from re import T
from matplotlib.colors import hsv_to_rgb
import math
import enviroment as env
import label
import stft
import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import PIL

env.override()


def loadData():
    dirPath = '/workspace/training/KEC/'
    index = os.listdir(dirPath)

    filePath = dirPath+random.choice(index)+'/speaker0.wav'
    print(filePath)
    wav, sr = librosa.load(filePath, mono=True, sr=env.sampleRate)
    return wav


def loadRoom():
    dirPath = '/workspace/training/KEC/'
    index = os.listdir(dirPath)

    filePath = dirPath+random.choice(index)+'/room.wav'
    print(filePath)
    wav, sr = librosa.load(filePath, mono=False, sr=env.sampleRate)
    print(wav.shape)
    return wav


def transform():
    print(stft.transform(loadData()))


def libSpec():
    y = loadData()
    fig, ax = plt.subplots()
    D_highres = stft.transform(y)
    rows = stft.freqBins()

    print(len(D_highres))

    print(len(rows))
    print(rows)
    cols = stft.frame_to_sample(len(D_highres[0]))
    secs = stft.samples_to_time(cols)
    print(cols)
    print(secs)

    S_db_hr = stft.to_dB(D_highres)
    img = librosa.display.specshow(S_db_hr, cmap='gray_r', hop_length=env.hop_Length, x_axis='time', y_axis='linear',
                                   ax=ax, sr=env.sampleRate)
    ax.set(title='Higher time and frequency resolution')
    fig.colorbar(img, ax=ax, format="%+2.f dB")


def libSpec2():
    y = loadData()
    D_highres = stft.transform(y)
    S_db = stft.to_dB(D_highres)
    S_ph = stft.to_phase(D_highres)
    # Construct a subplot grid with 3 rows and 1 column, sharing the x-axis)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    # On the first subplot, show the original spectrogram
    img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                    ax=ax[0], hop_length=env.hop_Length, n_fft=env.n_fft, sr=env.sampleRate)
    ax[0].set(title='STFT (log scale)')

    # On the second subplot, show the mel spectrogram
    img2 = librosa.display.specshow(S_ph, x_axis='time', y_axis='linear',
                                    ax=ax[1], hop_length=env.hop_Length, n_fft=env.n_fft, sr=env.sampleRate)
    ax[1].set(title='Mel')

    # To eliminate redundant axis labels, we'll use "label_outer" on all subplots:
    for ax_i in ax:
        ax_i.label_outer()

    # And we can share colorbars:
    fig.colorbar(img1, ax=[ax[0], ax[1]])


def libSpecPh(room=None):
    room = room if not room is None else loadRoom()
    [y1, y2] = room
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)
    ipd = stft.to_ipd(d1, d2)
    fig, ax = plt.subplots(dpi=200)
    img = librosa.display.specshow(ipd, cmap='hsv', hop_length=env.hop_Length, x_axis='time', y_axis='linear',
                                   ax=ax, sr=env.sampleRate)
    ax.set(title='Phase only')


def libSpec3(room=None):
    room = room if not room is None else loadRoom()
    [y1, y2] = room
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)
    ipd = stft.to_ipd(d1, d2) / (math.pi*2)

    iid = stft.to_iid(d1, d2) if env.iid else np.ones(d1.shape)

    intensity = np.maximum(np.abs(d1),np.abs(d2))
    feature = np.dstack((ipd, iid, intensity))

    print(feature[:, :, 0])
    plt.figure(dpi=200)
    plt.imshow(hsv_to_rgb(feature), origin='lower', extent=(
        0, stft.frame_to_time(len(d1[0])), 0, stft.freqBins()[-1]), aspect='auto',interpolation='none')
    plt.xlabel("t [s]")
    plt.ylabel("f [Hz]")
    plt.title("IA-STFT")
    plt.show()


def testLable():
    print(np.round( label.classifyOneHot(0),3))
    print(np.round( label.simpleClassify(0),3))
    print(np.round( label.simpleOneHot(0),3))

    print(np.round(label.classifyOneHot(math.pi),3))
    print(np.round(label.simpleClassify(math.pi),3))
    print(np.round(label.simpleOneHot(math.pi),3))

    print(np.round(label.classifyOneHot(-math.pi),3))
    print(np.round(label.simpleClassify(-math.pi),3))
    print(np.round(label.simpleOneHot(-math.pi),3))

    print(np.round(label.classifyOneHot(-math.pi/2),3))
    print(np.round(label.simpleClassify(-math.pi/2),3))
    print(np.round(label.simpleOneHot(-math.pi/2),3))

    print(np.round(label.classifyOneHot(+math.pi/2),3))
    print(np.round(label.simpleClassify(+math.pi/2),3))
    print(np.round(label.simpleOneHot(+math.pi/2),3))

    print(np.round(label.classifyOneHot(-(2/9)*math.pi),3))
    print(np.round(label.simpleClassify(-(2/9)*math.pi),3))
    print(np.round(label.simpleOneHot(-(2/9)*math.pi),3))

def testPil():
    [y1, y2] = loadRoom()
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)
    print(d1.shape)
    ipd = stft.to_ipd(d1, d2) / (math.pi*2)

    iid = stft.to_iid(d1, d2) if env.iid else np.ones(d1.shape)

    intensity = np.maximum(np.abs(d1),np.abs(d2))
    feature = np.dstack((ipd, iid, intensity))

    img = PIL.Image.fromarray(feature,mode='HSV')
    return img

