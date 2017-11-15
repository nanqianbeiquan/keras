# -*- coding: utf-8 -*-

from __future__ import print_function
from dataset.load_digit_dataset import load_dataset
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

path_name = 'E:\pest1\mnist'

images, labels = load_dataset(path_name)

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size = 0.3)

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))


model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(),
              metrics = ['accuracy'])

history = model.fit(x_train,y_train,
                    batch_size = 128,
                    epochs = 10,
                    verbose = 1,
                    validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy', score[1])

model.save('mnist.model.h5')






