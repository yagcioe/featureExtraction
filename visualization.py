from colorsys import hsv_to_rgb
from matplotlib import pyplot as plt
import stft
import librosa 
import enviroment as env

def iastft(feature):
    plt.figure(dpi=200)
    plt.imshow(hsv_to_rgb(feature), origin='lower', extent=(
        0, stft.frame_to_time(len(feature[0])), 0, stft.freqBins()[-1]), aspect='auto',interpolation='none')
    plt.xlabel("t [s]")
    plt.ylabel("f [Hz]")
    plt.title("IA-STFT")
    plt.show()

def phase(ipd):
    fig, ax = plt.subplots(dpi=200)
    img = librosa.display.specshow(ipd, cmap='hsv', hop_length=env.hop_Length, x_axis='time', y_axis='linear',
                                   ax=ax, sr=env.sampleRate)
    ax.set(title='Phase only')