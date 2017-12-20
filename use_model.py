import os
from keras.utils import to_categorical
import keras
from keras.models import model_from_json
import numpy as np
import librosa

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#print("data")
mfcc = wav2mfcc('/media/parth/06C20E27C20E1B95/deep learning/speechrecognizer/train/audio/dog/1a4259c3_nohash_1.wav')
mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
print(np.argmax(loaded_model.predict(mfcc_reshaped)))
print(loaded_model.predict(mfcc_reshaped))
