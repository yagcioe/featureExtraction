
import os
import random
import shutil
import librosa
from matplotlib.font_manager import json_load
import numpy as np
import enviroment as env
import feature
import label


def softmax(list):
    sum = np.sum(np.exp(list))
    return [np.exp(x)/sum for x in list]


def splitIndex(split, len):
    split = softmax(split)
    return [np.round(sum(split[0:i]) * len) for i in range(len(split))]


def run():
    allSamples = os.listdir(env.target_dir)
    random.shuffle(allSamples)
    idx = splitIndex(env.split, len(allSamples))

    transformSet(allSamples[0:idx[0]], 'train/')
    transformSet(allSamples[idx[0]:idx[1]], 'val/')
    transformSet(allSamples[idx[1]:idx[2]], 'test/')


def transformSet(allPAths, target):
    pathToExport = env.target_dir+target
    shutil.rmtree(pathToExport, True)
    for s in allPAths:
        fet, lab, id = transformSample(s)


        imgPath= pathToExport+'images/'
        os.makedirs(imgPath)
        feature.exportFeature(fet, imgPath+f'{id}/')

        labelPath=pathToExport+'labels/'
        os.makedirs(labelPath)
        label.exportLabel(lab, labelPath+f'{id}/')


def transformSample(s):
    wav = librosa.load(s+'room.wav')
    fet = feature.feature(wav)

    js = json_load(s+'description.json')
    lab = label.label(js)

    id = js.sample.id
    return fet, lab, id


if __name__ == '__main__':
    run()
