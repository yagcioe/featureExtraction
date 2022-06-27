import math
from PIL import Image
import numpy as np
import numpy as np
from matplotlib.colors import hsv_to_rgb

from . import enviroment as env
from . import stft


def feature(wav):
    [y1, y2] = wav
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)

    ipd = ipdFeature(d1, d2)
    iid = iidFeature(d1, d2)
    intensity = intensityFeature(d1, d2)

    feature = np.dstack((ipd, iid, intensity))

    feature = hsv_to_rgb(feature)[1:, 1:-1]
    # rm 0 freq and 1 and last time col to get 640x512 px
    iFeature = np.flip(np.floor(feature*256).astype(np.uint8), 0)
    fet = Image.fromarray(iFeature, 'RGB')
    return fet


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


def exportFeature(fet, name, path):
    stft.ensurePath(path)
    fullPath = path+name
    fet.save(fullPath)
