# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


X = np.linspace(-1, 1, 200)
np.random.shuffle(X)

Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,))

#plt.scatter(X,Y)
#plt.show()

X_train,Y_train = X[:160],Y[:160]
X_test, Y_test = X[160:],Y[160:]


model = Sequential()

model.add(Dense(output_dim=1,input_dim=1))
model.compile(loss = 'mse', optimizer='sgd')

print('Training','-'*20)

for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print("train cost:", cost)
        
print('Testing','+'*20)

cost = model.evaluate(X_test,Y_test,batch_size=40)
print("test cost:",cost)
W,b = model.layers[0].get_weights()
print("weights=",W,"\nbiases=",b )
#print('X_test:',X_test)
Y_pred = model.predict(X_test)
#print('Y_pred:',Y_pred)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)

json_string = model.to_json()

model = model_from_json(json_string)
print(model)
