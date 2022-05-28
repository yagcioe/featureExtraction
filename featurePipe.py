
import os
import random
import shutil
import librosa
from matplotlib.font_manager import json_load
import numpy as np
import enviroment as env
import feature
import label


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
    pathToExport = env.target_dir+target
    shutil.rmtree(pathToExport)
    imgPath = pathToExport+'images/'
    labelPath = pathToExport+'labels/'
    os.makedirs(imgPath)
    os.makedirs(labelPath)

    for s in allPAths:
        fet, lab, id = transformSample(env.source_dir+s+"/")
        feature.exportFeature(fet, imgPath+f'{id}.png')
        label.exportLabel(lab, labelPath+f'{id}.txt')


def transformSample(s):
    wav, sr = librosa.load(s+'room.wav', sr=env.sampleRate, mono=False)
    fet = feature.feature(wav)

    js = json_load(s+'description.json')
    lab = label.label(js)

    id = js['sample']['id']
    return fet, lab, id


if __name__ == '__main__':
    run()
