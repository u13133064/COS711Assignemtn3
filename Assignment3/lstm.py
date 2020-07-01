import pickle
import numpy as np
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def normalize(minv,maxv,x):
    if(x< minv or x > maxv ):
        print(x)
        raise NameError('HiThere')
    return (x-minv)/(maxv-minv)
file = open('Processedx.pkl', 'rb')
data = pickle.load(file)
file.close()


file = open('Processedy.pkl', 'rb')
ydata = pickle.load(file)
file.close()

file = open('ProcessedT.pkl', 'rb')
tdata = pickle.load(file)
file.close()
x = np.array(data).astype(np.float)
y = np.array(ydata).astype(np.float)
t = np.array(tdata).astype(np.float)


y = y.reshape(-1,1)
maxy = max(y)
miny = min(y)
newy = []
for i in y:
    newy.append(normalize(miny,maxy,i))
#y = np.array(newy)
print(y)
print(max(y))

first = Input(shape=(121,6))
lstm = LSTM(45,return_sequences=False)(first)
second= Dense(60, activation='relu')(lstm)

output= Dense(1)(second)

model = Model(inputs=first, outputs=output)
# summarize layers
opt = Adam(learning_rate=0.001)
print(model.summary())
loss=tf.keras.losses.Huber(delta=100.0)

model.compile(loss=loss, optimizer=opt,metrics=[ tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(x, y,  batch_size=256,epochs=50, verbose=1)
predictions = model.predict(t)
np.savetxt("results.csv", predictions, delimiter=",",fmt='%f')



plt.plot(history.history['root_mean_squared_error'], label="root_mean_squared_error")
plt.legend(loc="upper right")
plt.show()