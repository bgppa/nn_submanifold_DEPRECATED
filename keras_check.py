# Just run a keras instance in order to ensure that, given the nn architecture
# and the dataset, a suitable optimization is ACTUALLY possible
# (otherwise we are searching for nothing)
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from datalib import X_dataset, y_dataset
#from keras.utils import to_categorical


X_train = np.copy(X_dataset)
y_train = np.copy(y_dataset)
#y_train = to_categorical(y_train)

n_cols = X_train.shape[1]
print("Input dimension: ", n_cols)
num_classes = 2 # It is understood that we classify labels 0-1

# create model
model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',  
    metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, validation_split = 0.,
    epochs=5000, batch_size=10, verbose=2)


scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
#int("Model summary: ", model.summary())
model.summary()
