import numpy as np
import librosa
import os

DATA_PATH = "./train/audio"


def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def preprocessing_data(path=DATA_PATH, max_pad_len=11):
    try:
        s_data = [data for data in os.listdir(path)]
    except:
        print("error in fetching labels list")
        sys.exit()

    for data in s_data:
        mfcc_vectors = []
        filepath = DATA_PATH + '/' + data
        for allfiles in os.listdir(filepath):
            mfcc = wav2mfcc(filepath+ '/' +allfiles)
            mfcc_vectors.append(mfcc)
        np.save("data_np" + '/' + data + '.npy', mfcc_vectors)
        print(data + ".npy  filesaved")

preprocessing_data()
