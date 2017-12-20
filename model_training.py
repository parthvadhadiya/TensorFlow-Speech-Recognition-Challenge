import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
DATA_PATH = "./train/audio"


def get_labels(data):
    labels = os.listdir(data)
    return labels

def prepare_training_data(data):
	labels = get_labels(data)
	X = np.load("data_np"+ '/' + labels[0] + '.npy')
	y = np.zeros(X.shape[0])

	for _, label in enumerate(labels[1:]):
		x = np.load("data_np"+ '/' + label + '.npy')
		X = np.vstack((X, x))
		y = np.append(y, np.full(x.shape[0], fill_value= (_ + 1)))
	assert X.shape[0] == len(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)
	return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_training_data(DATA_PATH)


X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(30, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train_one_hot, batch_size=100, epochs=400, verbose=1)

model_json = model.to_json()
with open("./model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("./model.h5")
print("saved model..! ready to go.")
score = model.evaluate(X_train, y_train_one_hot, verbose=1)
