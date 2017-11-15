# -*- coding: utf-8 -*-

import numpy as np

from dataset.load_digit_dataset import load_dataset
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input
from keras.models import Model
import matplotlib.pyplot as plt

path_name = 'E:\pest1\mnist'
images,labels = load_dataset(path_name)

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.3)

x_train = x_train.astype('float32')/255 - 0.5
x_test = x_test.astype('float32')/255 -0.5

x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)

print(x_train.shape)
print(x_test.shape)

encoding_dim = 2
input_img = Input(shape=(784,))
print(input_img)

encoded = Dense(128,activation='relu')(input_img)
encoded = Dense(64,activation = 'relu')(encoded)
encoded = Dense(10,activation = 'relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

decoded = Dense(10,activation = 'relu')(input_img)
decoded = Dense(64,activation = 'relu')(decoded)
decoded = Dense(128,activation = 'relu')(decoded)
decoded = Dense(784,activation = 'tanh')(decoded)

autoencoder = Model(input = input_img,output = decoded)
encoder = Model(input = input_img,output = encoder_output)

autoencoder.compile(optimizer='adam', loss = 'mse')
autoencoder.fit(x_train,x_train,
                epochs = 20,
                batch_size = 256,
                shuffle = True)

encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)

plt.colorbar()
plt.show()