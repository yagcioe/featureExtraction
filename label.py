
from matplotlib.font_manager import json_load
from numpy import argmax

import stft
import enviroment as env


def label(js):
    # load descriptor
    boundingBoxes = []
    for speaker in js['sample']['speakers']:
        startFrame = stft.time_to_frame(speaker['startTime'])
        endFrame = stft.time_to_frame(speaker['endTime'])
        width = stft.time_to_frame(speaker['duration'])

        heigth = len(stft.freqBins())

        x = (startFrame+endFrame)/2
        y = heigth/2 # always center on the middel of the picture

        clas = simpleClassify(speaker['direction']['azimuth'])

        box = stringifySimpleLabel([clas, x, y, width, heigth])

        boundingBoxes.append(box)
    return boundingBoxes

def exportLabel(lab,path):
    with  open(path,'w') as f:
        for l in lab:
            f.write(l+"\n")


def stringifySimpleLabel(lab):
    lab = list(map(str, lab))
    return '\t'.join(lab)

def stringifyLabel(lab):
    pass

def divideInterval(interval, n):
    [minimum, maximum] = interval
    diff = maximum-minimum
    step = diff/n
    return [(i*step)+minimum for i in range(n+1)]


def classifyOneHot(azimuth):
    """return array of classes. one class has a percwntage of 1 when azimuth is perfectly centered, otherwise the azimuth has some overlap in different classes """
    # size n+1 so that for loop is well defined
    intervalls = divideInterval(env.anlges_interval, env.azimuth_slice_count)
    step = (env.anlges_interval[1]-env.anlges_interval[0]
            )/env.azimuth_slice_count  # max-min/2
    classes = []
    for n in range(env.azimuth_slice_count):
        start = max(intervalls[n], azimuth-(step/2))
        end = min(intervalls[n+1], azimuth+(step/2))

        # if start>end then azimuth is not in this segment
        lengthInSegment = max(0, end-start)
        percentageInSegment = lengthInSegment/step
        classes.append(percentageInSegment)
    return classes


def simpleClassify(azimuth):
    return argmax(classifyOneHot(azimuth))


def simpleOneHot(azimuth):
    return [1 if i == simpleClassify(azimuth) else 0 for i in range(env.azimuth_slice_count)]
