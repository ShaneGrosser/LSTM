from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

yt = []
y_1 = 0
y_2 = 0
random.seed(2017)

for i in range(500):
    #e = round(random.random(), 2)
    e = 0
    y_i = 5 + 0.95 * y_1 - 0.18 * y_2 + e
    y_2 = y_1
    y_1 = y_i
    yt.append(y_i)
    i = i + 1

yt_1 = pd.Series(yt).shift(1)
yt_2 = pd.Series(yt).shift(2)
yt_3 = pd.Series(yt).shift(3)
yt_4 = pd.Series(yt).shift(4)
yt_5 = pd.Series(yt).shift(5)

data = pd.concat([pd.Series(yt), yt_1, yt_2, yt_3, yt_4, yt_5], axis=1)
data = data.dropna()

y = data.iloc[:, 0]
x = data.iloc[:, 1:]

scaler_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
x = np.array(x).reshape((len(x), 5))
x = scaler_x.fit_transform(x)

scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))
y = np.array(y).reshape((len(y), 1))
y = scaler_y.fit_transform(y)

train_end = 350

x_train = x[0:train_end]
x_test = x[train_end:500]
y_train = y[0:train_end]
y_test = y[train_end:500]

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

seed = 2016
num_epochs = 500
np.random.seed(seed)
fit = Sequential()
fit.add(LSTM(output_dim=5, return_sequences=False, activation='tanh', inner_activation='relu', input_shape=(1, 5)))
fit.add(Dense(output_dim=1, activation='linear'))
fit.compile(loss='mean_squared_error', optimizer='rmsprop')

fit.fit(x_train, y_train, batch_size=1, nb_epoch=num_epochs, shuffle=True)
pred = fit.predict(x_test)
pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred), 1)))
y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len(y_test), 1)))

pred2 = fit.predict(x_train)
pred2 = scaler_y.inverse_transform(np.array(pred2).reshape((len(pred2), 1)))
y_train = scaler_y.inverse_transform(np.array(y_train).reshape((len(y_train), 1)))

plt.plot(y_train)
plt.plot(pred2)
plt.show()

errors = y_test - pred

plt.plot(y_test)
plt.plot(pred)
plt.show()

df = pd.DataFrame(y_train,columns = ['Actuals'])
df['Predictions'] = pred2

df_test = pd.DataFrame(y_test,columns = ['Actuals'])
df_test['Predictions'] = pred