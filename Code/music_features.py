import random
import string
import librosa as librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment


def convert_mp3_to_wav(filepath):
    letters = string.ascii_lowercase
    name = ''.join(random.choice(letters) for i in range(5))
    wav_file_name = name + ".wav"
    wave_file_path = "Mov_Files/" + wav_file_name
    song = AudioSegment.from_mp3(filepath)
    song.export(wave_file_path, format="wav")
    return wav_features("Mov_Files/" + wav_file_name)


def generate_mov_wavelength(filepath):
    return librosa.load(filepath)


def wav_features(filepath):
    y, sr = generate_mov_wavelength(filepath)
    chroma_stft = np.array(librosa.feature.chroma_stft(y=y, sr=sr)[0]).flatten().mean()
    rmse = np.array(librosa.feature.rms(y=y)[0]).flatten().mean()
    spectral_centroid = np.array(librosa.feature.spectral_centroid(y, sr=sr)[0]).flatten().mean()
    spectral_bandwidth = np.array(librosa.feature.spectral_bandwidth(y=y, sr=sr)).flatten().mean()
    rolloff = np.array(librosa.feature.spectral_rolloff(y=y, sr=sr)).flatten().mean()
    zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y)).flatten().mean()
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr)
    mfcc1 = np.array(mfcc_features[0]).flatten().mean()
    mfcc2 = np.array(mfcc_features[1]).flatten().mean()
    mfcc3 = np.array(mfcc_features[2]).flatten().mean()
    mfcc4 = np.array(mfcc_features[3]).flatten().mean()
    mfcc5 = np.array(mfcc_features[4]).flatten().mean()
    mfcc6 = np.array(mfcc_features[5]).flatten().mean()
    mfcc7 = np.array(mfcc_features[6]).flatten().mean()
    mfcc8 = np.array(mfcc_features[7]).flatten().mean()
    mfcc9 = np.array(mfcc_features[8]).flatten().mean()
    mfcc10 = np.array(mfcc_features[9]).flatten().mean()
    mfcc11 = np.array(mfcc_features[10]).flatten().mean()
    mfcc12 = np.array(mfcc_features[11]).flatten().mean()
    mfcc13 = np.array(mfcc_features[12]).flatten().mean()
    mfcc14 = np.array(mfcc_features[13]).flatten().mean()
    mfcc15 = np.array(mfcc_features[14]).flatten().mean()
    mfcc16 = np.array(mfcc_features[15]).flatten().mean()
    mfcc17 = np.array(mfcc_features[16]).flatten().mean()
    mfcc18 = np.array(mfcc_features[17]).flatten().mean()
    mfcc19 = np.array(mfcc_features[18]).flatten().mean()
    mfcc20 = np.array(mfcc_features[19]).flatten().mean()

    data = pd.DataFrame(
        {"chroma_stft": chroma_stft, "rmse": rmse, "spectral_centroid": spectral_centroid,
         "spectral_bandwidth": spectral_bandwidth, "rolloff": rolloff,
         "zero_crossing_rate": zero_crossing_rate,
         "mcff1": mfcc1,
         "mcff2": mfcc2,
         "mcff3": mfcc3,
         "mcff4": mfcc4,
         "mcff5": mfcc5,
         "mcff6": mfcc6,
         "mcff7": mfcc7,
         "mcff8": mfcc8,
         "mcff9": mfcc9,
         "mcff10": mfcc10,
         "mcff11": mfcc11,
         "mcff12": mfcc12,
         "mcff13": mfcc13,
         "mcff14": mfcc14,
         "mcff15": mfcc15,
         "mcff16": mfcc16,
         "mcff17": mfcc17,
         "mcff18": mfcc18,
         "mcff19": mfcc19,
         "mcff20": mfcc20,
         }, index=[0]
    )
    return data


def print_label(prediction):
    if prediction == 0:
        return "The Music is a Blues Genre"
    if prediction == 1:
        return "The Music is a Classical Genre"
    if prediction == 2:
        return "The Music is a Country Genre"
    if prediction == 3:
        return "The Music is a Disco Genre"
    if prediction == 4:
        return "The Music is a Hip-Hop Genre"
    if prediction == 5:
        return "The Music is a Jazz Genre"
    if prediction == 6:
        return "The Music is a Metal Genre"
    if prediction == 7:
        return "The Music is a Pop Genre"
    if prediction == 8:
        return "The Music is a Reggae Genre"
    if prediction == 9:
        return "The Music is a Rock Genre"
