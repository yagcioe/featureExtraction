
import numpy as np
from . import stft
from . import enviroment as env


def label(js):
    # load descriptor
    boundingBoxes = []
    duration = stft.time_to_frame(js['sample']['duration'])
    for speaker in js['sample']['speakers']:
        startFrame = stft.time_to_frame(speaker['startTime'])
        endFrame = stft.time_to_frame(speaker['endTime'])
        width = stft.time_to_frame(speaker['duration'])

        heigth = 1

        x = (startFrame+endFrame)/2

        x = x / duration
        width = width / duration
        y = 0.5  # always center on the middel of the picture

        azimuth = speaker['direction']['azimuth']
        clas = simpleClassify(azimuth)
        box = [clas, x, y, width, heigth]  # normalize to [0,1]

        boundingBoxes.append(box)
    return boundingBoxes


def exportLabel(lab, name, path):
    stft.ensurePath(path)
    with open(path+name, 'w') as f:
        for l in lab:
            f.write(stringifySimpleLabel(l)+"\n")


def stringifySimpleLabel(lab):
    lab = list(map(str, lab))
    return ' '.join(lab)


def stringifyLabel(lab):
    pass


def divideInterval(interval, n):
    diff = interval[1]-interval[0]
    fr = diff/n
    sections = []
    for i in np.arange(-(n/2), (n/2), 1):
        sections.append(np.array([i, i+1]))
    sections = np.array(sections)
    sections = fr*sections
    return sections, fr


def classifyOneHot(azimuth):
    """return array of classes. one class has a percwntage of 1 when azimuth is perfectly centered, otherwise the azimuth has some overlap in different classes """
    intervalls, step = divideInterval(
        env.anlges_interval, env.azimuth_slice_count)
    classes = []
    low = azimuth-(step/2)
    high = azimuth+(step/2)
    for n, intv in enumerate(intervalls):
        if (low >= intv[0] and low <= intv[1]) or \
                (high >= intv[0] and high <= intv[1]):
            # wenn anfang oder ende des step intevall um azimuth in der beobachteten section ist

            start = max(intv[0], low)
            end = min(intv[1], high)

            lengthInSegment = end-start
            percentageInSegment = lengthInSegment/step
            classes.append(percentageInSegment)
        else:
            classes.append(0)
    return classes


def simpleClassify(azimuth):
    return np.argmax(classifyOneHot(azimuth))


def simpleOneHot(azimuth):
    return [1 if i == simpleClassify(azimuth) else 0 for i in range(env.azimuth_slice_count)]
