import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = web.DataReader('GC=F' ,data_source='yahoo' ,start='2017-01-01' ,end='2022-08-01')
data.shape
data

plt.style.use('fivethirtyeight')
plt.title('Fiyatlar')
plt.ylabel('Dolar')
data['Close'].plot(figsize=(25,15))
plt.show()

data = data[['Close']]
data.tail()

tahminGun = 20
data['Tahmin'] = data[['Close']].shift(-tahminGun)
data.tail()

X = np.array(data.drop(['Tahmin'],1))[:-tahminGun]
y = np.array(data['Tahmin'])[:-tahminGun]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
model1 = LinearRegression().fit(X_train, y_train)
model2 = DecisionTreeRegressor().fit(X_train, y_train)

X_tahmin = data.drop(['Tahmin'],1)[-tahminGun:]
X_tahmin = X_tahmin.tail(tahminGun)
X_tahmin = np.array(X_tahmin)

model1_tahmin = model1.predict(X_tahmin)

model2_tahmin = model2.predict(X_tahmin)

valid = data[X.shape[0]:]
valid['Tahmin'] = model1_tahmin
plt.figure(figsize=(25,12))
plt.xlabel('Gunler')
plt.ylabel('Fiyat')
plt.plot(data['Close'])
plt.plot(valid[['Close', 'Tahmin']])
plt.legend(['Orjinal', 'Deger' ,'Tahmin'])
plt.show()


valid = data[X.shape[0]:]
valid['Tahmin'] = model2_tahmin
plt.figure(figsize=(25,12))
plt.xlabel('Gunler')
plt.ylabel('Fiyat')
plt.plot(data['Close'])
plt.plot(valid[['Close', 'Tahmin']])
plt.legend(['Orjinal', 'Deger' ,'Tahmin'])
plt.show()
