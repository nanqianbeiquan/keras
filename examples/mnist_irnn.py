# -*- coding: utf-8 -*-

from keras.layers import Dense, Flatten, Conv2D
from keras.layers import SimpleRNN
from keras.utils import np_utils
from keras.models import Sequential
from keras import initializers
from keras.optimizers import RMSprop

from dataset.load_digit_dataset import load_dataset
from sklearn.model_selection import train_test_split

batch_size = 32
epochs = 10
hidden_units = 100
learning_rate = 1e-6
clip_norm = 1.0

path_name = 'E:\pest1\mnist'

images, labels = load_dataset(path_name)

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.3)
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

model = Sequential()

model.add(SimpleRNN(hidden_units,
                    kernel_initializer = initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer = initializers.Identity(gain=1.0),
                    activation = 'relu',
                    input_shape = x_train.shape[1:]))

model.add(Dense(10,activation = 'softmax'))
rmsprop = RMSprop(lr = learning_rate)
model.compile(loss = 'categorical_crossentropy',
              optimizer = rmsprop,
              metrics = ['accuracy'])

model.summary()

model.fit(x_train,y_train,
          epochs = epochs,
          batch_size = batch_size,
          verbose = 1,
          validation_data = (x_test,y_test))

score = model.evaluate(x_test,y_test,varbose=0)

print('IRNN test score:', score[0])
print('IRNN test acc:',score[1])
