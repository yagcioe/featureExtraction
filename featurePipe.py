
import os
import random
import shutil
import librosa
from matplotlib.font_manager import json_load
import numpy as np
from . import enviroment as env
from . import feature
from . import label

env.override()

def perc(list):
    sum = np.sum(list)
    return [x/sum for x in list]


def splitIndex(split, n):
    split = perc(split)
    return [int(np.round(sum(split[0:i]) * n)) for i in range(len(split))]


def run():
    allSamples = os.listdir(env.source_dir)
    random.shuffle(allSamples)
    idx = splitIndex(env.split, len(allSamples))
    print('start train')
    transformSet(allSamples[idx[0]:idx[1]], 'train/')
    print('start val')

    transformSet(allSamples[idx[1]:idx[2]], 'val/')
    print('start test')

    transformSet(allSamples[idx[2]:len(allSamples)], 'test/')
    print('done')


def transformSet(allPAths, target):
    shutil.rmtree(env.target_dir+target)
    for s in allPAths:
        fet, lab, id = transformPath(s)
        export(fet, lab, id, target)


def export(fet, lab, id, target):
    feature.exportFeature(fet, f'{id}.png', env.target_dir+target+'images/')
    label.exportLabel(lab, f'{id}.txt', env.target_dir+target+'labels/')

def transformPath(s):
    wav, js = loadSample(s)
    return transform(wav, js)


def loadSample(path):
    wav, sr = librosa.load(path+'room.wav', sr=env.sampleRate, mono=False)
    js = json_load(path+'description.json')
    return wav, js


def transform(wav, js):
    fet = feature.feature(wav)
    lab = label.label(js)
    id = js['sample']['id']
    return fet, lab, id

def clearDir():
    shutil.rmtree(env.target_dir)


if __name__ == '__main__':
    run()
