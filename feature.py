from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import stft
import math
import numpy as np
import enviroment as env


def feature(wav):
    [y1, y2] = wav
    d1 = stft.transform(y1)
    d2 = stft.transform(y2)
    ipd = stft.to_ipd(d1, d2) / (math.pi*2)

    iid = stft.to_iid(d1, d2) if env.iid else np.ones(d1.shape)

    intensity = np.maximum(np.abs(d1),np.abs(d2))
    feature = np.dstack((ipd, iid, intensity))
    
    fig = plt.figure()
    plt.imshow(hsv_to_rgb(feature), origin='lower',interpolation='none')
    plt.axes('off')
    return fig

def exportFeature(fig, path):
    plt.figure(fig)
    plt.axes('off')
    plt.savefig(path,bbox_inches='tight')
