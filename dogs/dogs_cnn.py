# coding=utf-8

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from load_dog_dataset import load_dataset
import keras
import numpy as np

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_dogs_trained_model.h5'

x_train, y_train,x_test, y_test, label_map = load_dataset()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train=np.asarray(x_train,dtype=float)
print(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_train.shape[1:])
print(x_train.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(3,28,28)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(x_train, y_train,
		  batch_size=32,
		  epochs=1,
		  validation_data=(x_test, y_test),
		  shuffle=True)
score = model.evaluate(x_test, y_test, batch_size=32)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255