from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

"""READ DATA"""
df = pd.read_excel('P:/Coding/Modelling/IUK/IUK_data_PCA.xlsx')
df = df['2015-10-01':]

predictors = ['Basis', 'NBP/TTF', 'NCG/TTF', 'VTP/TTF', 'PEGN/TTF', 'PSV/TTF', 'UK T',
              'UK Norm T', 'DE T', 'DENorm T', 'BBL Rolloff', 'Exit', 'Entry', 'W', 'Q2', 'Q3', 'Hedge']

df_x = df[predictors]
df_y = df['IUK']

"""SCALE DATA"""
scaler_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
array_x = np.array(df_x).reshape((len(df_x), df_x.columns.shape[0]))
array_x = scaler_x.fit_transform(array_x)
scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))
array_y = np.array(df_y).reshape((len(df_y), 1))
array_y = scaler_y.fit_transform(array_y)
df_x_scaled = pd.DataFrame(array_x)
df_y_scaled = pd.DataFrame(array_y)

"""RESHAPE DATA"""
timesteps = 3
x = []

for i in range(timesteps):
    df_dum = df_x_scaled.shift(i)
    x.append(np.array(df_dum[timesteps:]))

y = np.array(df_y_scaled)[timesteps:]

x = np.array(x)
x = np.reshape(x, (x.shape[1], x.shape[0], x.shape[2]))

train_end = 350

x_train = x[0:train_end]
x_test = x[train_end:502]
y_train = y[0:train_end]
y_test = y[train_end:502]

seed = 2016
num_epochs = 10
np.random.seed(seed)
fit = Sequential()
fit.add(
    LSTM(output_dim=7, return_sequences=False, activation='tanh', inner_activation='relu', input_shape=(timesteps, 17)))
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

plt.plot(errors)
plt.show()