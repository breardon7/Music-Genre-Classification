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


def wav_features(filepath):
    y, sr = librosa.load(filepath)
    chroma_stft = np.array(librosa.feature.chroma_stft(y=y, sr=sr)[0])
    rmse = np.array(librosa.feature.rms(y=y)[0])
    spectral_centroid = np.array(librosa.feature.spectral_centroid(y, sr=sr)[0])
    spectral_bandwidth = np.array(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.array(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y))
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr)
    data = pd.DataFrame(
        {"chroma_stft": chroma_stft.flatten(), "rmse": rmse.flatten(), "spectral_centroid": spectral_centroid.flatten(),
         "spectral_bandwidth": spectral_bandwidth.flatten(), "rolloff": rolloff.flatten(),
         "zero_crossing_rate": zero_crossing_rate.flatten(),
         "mcff1": np.array(mfcc_features[0]).flatten(),
         "mcff2": np.array(mfcc_features[1]).flatten(),
         "mcff3": np.array(mfcc_features[2]).flatten(),
         "mcff4": np.array(mfcc_features[3]).flatten(),
         "mcff5": np.array(mfcc_features[4]).flatten(),
         "mcff6": np.array(mfcc_features[5]).flatten(),
         "mcff7": np.array(mfcc_features[6]).flatten(),
         "mcff8": np.array(mfcc_features[7]).flatten(),
         "mcff9": np.array(mfcc_features[8]).flatten(),
         "mcff10": np.array(mfcc_features[9]).flatten(),
         "mcff11": np.array(mfcc_features[10]).flatten(),
         "mcff12": np.array(mfcc_features[11]).flatten(),
         "mcff13": np.array(mfcc_features[12]).flatten(),
         "mcff14": np.array(mfcc_features[13]).flatten(),
         "mcff15": np.array(mfcc_features[14]).flatten(),
         "mcff16": np.array(mfcc_features[15]).flatten(),
         "mcff17": np.array(mfcc_features[16]).flatten(),
         "mcff18": np.array(mfcc_features[17]).flatten(),
         "mcff19": np.array(mfcc_features[18]).flatten(),
         "mcff20": np.array(mfcc_features[19]).flatten(),
         }
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
