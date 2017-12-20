import os
import keras
from keras.models import model_from_json
import numpy as np
import librosa
from tqdm import tqdm

path = "./test/test/audio"

def getlabel(choice):
	ans = ''
	my_list = ['bed','bird','cat','dog','down','eight','five','four','go','happy','house','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','wow','yes','zero']
	ans = str(my_list[choice])
	return ans

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
print("model loaded")


with open('sample_submission.csv','a') as f:
	columnTitleRow = "fname,label\n"
	f.write(columnTitleRow)

with open('sample_submission.csv','a') as f:
	for data in tqdm(os.listdir(path)):
		mfcc = wav2mfcc(path+'/'+data)
		mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
		result = np.argmax(loaded_model.predict(mfcc_reshaped))
		ans = getlabel(result)
		row = '{},{}\n'.format(data,ans)
		f.write(row)

