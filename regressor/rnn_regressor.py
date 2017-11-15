# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM,Dense,TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop

BATCH_SIZE = 50
BATCH_START = 0
TIME_STEPS = 2

def get_batch():
    global BATCH_START,TIME_STEPS
    xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE)
    xs = xs.reshape((BATCH_SIZE,TIME_STEPS))//(10*np.pi)

    seq = np.sin(xs)
    res = np.cos(xs)

#    plt.scatter(seq,res)
    BATCH_START += TIME_STEPS
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

model = Sequential() 
model.add(LSTM(50,
               input_shape = (TIME_STEPS,1),
               batch_size = 50,
               return_sequences =True,
               stateful = True)) 

model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='rmsprop')
X_batch,Y_batch, xs = get_batch()

for step in range(501):
#    X_batch,Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch,Y_batch)
    pred = model.predict(X_batch,BATCH_SIZE)
    plt.plot(xs[0,:],Y_batch[0].flatten(),'r',pred.flatten()[:TIME_STEPS],'b--')
    plt.ylim((-1.2,1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost:',cost)
    
if __name__ == '__main__':
    get_batch()
