import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "C:\\Users\\kui45\\"
data = pd.read_csv(path + '02-1_dataset.csv')

x = data['X'].values
noise = np.random.randn(len(x))
y = 5*x + (50*noise)

def e_mse(t):
    diff = (t - y)**2
    e_mse = diff.sum() / len(y)
    return e_mse

w = 0.5
b = 0.5
x0 = 0
x1 = 100

def h(x,w,b):
    return w*x+b

plt.figure()
plt.scatter(x,y)
plt.plot([x0, x1], [h(x0,w,b), h(x1,w,b)], 'r')

plt.text(20, 400, "MSE Loss: %.2f"%e_mse(h(x, w, b)))

learning_iter = 100
learning_rate = 0.0001

for i in range(learning_iter):
    error = (h(x,w,b)-y)
    w -= learning_rate * (error*x).sum()/len(x)
    b -= learning_rate * error.sum()
    y_hat = h(x, w, b)
    e_mse_new = e_mse(y_hat)

plt.figure()
plt.scatter(x,y)
plt.plot([x0, x1], [h(x0,w,b), h(x1,w,b)], 'r')
plt.text(20, 400, "MSE Loss: %.2f"%e_mse_new)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

x_reshaped = x[:, np.newaxis]
regr = linear_model.LinearRegression()
regr.fit(x_reshaped, y)
y_pred = regr.predict(x_reshaped)

e_mse_sklearn = mean_squared_error(y, y_pred)

plt.figure()
plt.scatter(x,y)
plt.plot(x_reshaped, y_pred, 'r')
plt.text(20, 400, "MSE Loss: %.2f"%e_mse_sklearn)
