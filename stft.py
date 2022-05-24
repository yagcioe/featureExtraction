import librosa


def transform(wav:list[float]):
    #Todo add sr
    return librosa.stft(wav)