import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

import featureExtraction.enviroment as env
import featureExtraction.stft as stft


def feature(wav):
    [y1, y2] = wav
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)

    ipd = ipdFeature(d1, d2)
    iid = iidFeature(d1, d2)
    intensity = intensityFeature(d1, d2)

    # print(intensity)
    # print(intensity.shape)
    # print(np.max(intensity))
    # idx = np.unravel_index(np.argmax(intensity), intensity.shape)
    # f , t = idx
    # print(idx)

    # print(intensity[:,t])
    feature = np.dstack((ipd, iid, intensity))

    fig = plt.figure(frameon=False)
    feature = hsv_to_rgb(feature)
    plt.imshow(feature, origin='lower', interpolation='none', aspect= 'auto')
    plt.axis('off')
    plt.close(fig)
    return fig


def ipdFeature(D, E):
    diff = stft.to_phase(D)-stft.to_phase(E)
    return (boundAngle(diff) / (math.pi*2))


def boundAngle(phi):
    return phi % (2*math.pi)


def intensityFeature(D, E):
    intensity = np.maximum(np.abs(D), np.abs(E))
    intensity = stft.to_dB(intensity)

    low = np.min(intensity)
    high = np.max(intensity)
    diff = high-low
    intensity = (intensity-low) / diff
    return intensity


def iidFeature(D, E):
    return stft.to_iid(D, E) if env.iid else np.ones(D.shape)


def exportFeature(fig,name ,path):
    stft.ensurePath(path)
    fullPath=path+name
    plt.figure(fig)
    plt.axis('off')
    plt.savefig(fullPath, bbox_inches='tight',pad_inches=0)
    plt.close(fig)
