# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:19:44 2020

@author: jedd.shneier
"""


"""
Each sample is a series of 121 data points of(6/7) varables. So as a 1 d array we can take each day as a seperate block
"""
from keras.utils import plot_model

import gc
import pickle
import numpy as np
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense
from keras.models import Sequential, Model
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def normalize(minv,maxv,x):
    if(x< minv or x > maxv ):
        print(x)
        raise NameError('HiThere')
    return (x-minv)/(maxv-minv)


def GetData():
    file = open('Processedx.pkl', 'rb')
    data = pickle.load(file)
    file.close()


    file = open('Processedy.pkl', 'rb')
    ydata = pickle.load(file)
    file.close()
    return data, ydata
def CreateModel():
     model = Sequential()
     model.add(Conv1D(64, kernel_size=11, activation='relu', input_shape=(121,7)))
     model.add(MaxPooling1D(3))
     model.add(Conv1D(32, kernel_size=5, activation='relu'))
     model.add(MaxPooling1D(3))
     model.add(Flatten())
     model.add(Dense(1))
     return model
 
def CreateGRU():
     first = Input(shape=(121,7))
     lstm = GRU(45,return_sequences=False, name="Recurrrent_layer" )(first)
     second= Dense(60, activation='relu')(lstm)

     output= Dense(1)(second)

     model = Model(inputs=first, outputs=output)
     return model
def CreateLSTM():
     first = Input(shape=(121,7))
     lstm = LSTM(45,return_sequences=False, name="Recurrrent_layer" )(first)
     second= Dense(60, activation='relu')(lstm)

     output= Dense(1)(second)

     model = Model(inputs=first, outputs=output)
     return model
def CreateRNN():
     first = Input(shape=(121,7))
     lstm = SimpleRNN(45,return_sequences=False, name="Recurrrent_layer" )(first)
     second= Dense(60, activation='relu')(lstm)

     output= Dense(1)(second)

     model = Model(inputs=first, outputs=output)
     return model
def PrepData(x,y):
    x = np.array(x)
    y = np.array(y).astype(np.float)
    y = y.reshape(-1,1)
    maxy = max(y)
    miny = min(y)
    newy = []
    for i in y:
        newy.append(normalize(miny,maxy,i))
    #y = np.array(newy)
    return train_test_split(x, y, test_size=0.33, random_state=1)
    
x,y = GetData()
#Uncomment model to test
model = CreateLSTM()
#model = CreateRNN()
#model = CreateGRU()

#model =  CreateModel()
#Change file Name for model
plot_model(model, to_file='modelCNN.png', show_shapes=True,show_layer_names=False)

trainX,testX,trainY,testY = PrepData(x,y)
print(testX[0])
print(testY.shape)
print(trainY.shape)

print(trainX.shape)
loss=tf.keras.losses.Huber(delta=100.0)
opt = Adam(learning_rate=0.001)
res = np.array([])
for i in range(10):
    tf.keras.backend.clear_session()
    model =  CreateModel()



    model.compile(loss=loss, optimizer=opt,metrics=[ tf.keras.metrics.RootMeanSquaredError()])
    #change bactch size and epochs
    history = model.fit(trainX,trainY,  batch_size=1024,epochs=100, verbose=1)
    _, avgRME = model.evaluate(testX, testY, batch_size=128)
    res = np.append(res, avgRME)
    plt.plot(history.history['root_mean_squared_error'], label="root_mean_squared_error")
plt.legend(loc="upper right")
plt.show()
print(np.mean(res))
print(np.std(res))
#modelLstm.compile(loss=loss, optimizer=opt,metrics=[ tf.keras.metrics.RootMeanSquaredError()])
#history = modelLstm.fit(trainX,trainY,  batch_size=256,epochs=100, verbose=1)


#results = modelLstm.evaluate(testX, testY, batch_size=256)
#print("test loss, test acc:", results)
gc.collect()
