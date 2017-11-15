# -*- coding: utf-8 -*-

from dataset.load_digit_dataset import load_dataset
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

path_name = 'E:\pest1\mnist'
images,labels=load_dataset(path_name)

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.3)

x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

print(x_train[1].shape)
print(y_train[:3])

model = Sequential()

model.add(Dense(32,input_shape=(784,),activation = 'relu'))
model.add(Dense(10,activation='softmax'))

rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer = rmsprop,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

print('Training','-'*20)
model.fit(x_train,y_train,epochs=20,batch_size=32)
loss,accuracy = model.evaluate(x_test,y_test,verbose=0)

print('loss:',loss)
print('accuracy:',accuracy)