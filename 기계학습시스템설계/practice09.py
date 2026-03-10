import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path = "C:\\Users\\kui45\\"
data = pd.read_csv(path + '03-2_dataset.csv')

x = data['X'].values
x = x.reshape(-1, 1)

plt.figure()
plt.plot(x)

plt.figure()
plt.hist(x, bins=100)

def normalization(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x_normal = (x - x_min) / (x_max - x_min)
    return x_max, x_min, x_normal

x_max, x_min, x_normal = normalization(x)

x_n_max = np.max(x_normal)
x_n_min = np.min(x_normal)

scaler = MinMaxScaler()
n_x = scaler.fit_transform(x)

n_x_max = np.max(n_x)
n_x_min = np.min(n_x)

plt.figure()
plt.plot(x_normal)

plt.figure()
plt.hist(x_normal, bins=100)

print("원본데이터의 최대값 : {:.5f}  최소값 : {:.5f}".format(x_max,x_min))
print("정규화 된 데이터의 최대값 : {:.5f}  최소값 : {:.5f}".format(x_n_max,x_n_min))
print("scikit_learn 정규화 데이터의 최대값 : {:.1f}  최소값 : {:.1f}".format(n_x_max, n_x_min))

def standard(x):
    x_mean = x.sum() / len(x)
    x_var = ((x - x_mean)**2).sum() / len(x)
    x_std = np.sqrt(x_var)
    x_dev = (x - x_mean) / x_std
    return x_mean, x_std, x_dev

x_mean, x_std, x_dev = standard(x)

x_s_mean = np.mean(x_dev)
x_s_std = np.std(x_dev)

scaler_a = StandardScaler()
s_x = scaler_a.fit_transform(x)

s_x_mean = np.mean(s_x)
s_x_std = np.std(s_x)

plt.figure()
plt.plot(x_dev)

plt.figure()
plt.hist(x_dev, bins=100)

print("원본데이터의 평균값 : {:.2f}  표준편차 : {:.2f}".format(x_mean,x_std))
print("정규화 된 데이터의 평균값 : {:.2f}  표준편차 : {:.2f}".format(x_s_mean,x_s_std))
print("scikit_learn 정규화 데이터의 평균값 : {:.2f}  표준편차 : {:.2f}".format(s_x_mean, s_x_std))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

noise = np.random.randn(len(x)).reshape(-1, 1)
n_y = (-2*n_x) + 1 + (1.2*noise)

n_x_train,n_x_test,n_y_train,n_y_test = train_test_split(n_x, n_y, test_size=0.2)

regr = LinearRegression()
regr.fit(n_x_train, n_y_train)
n_y_pred_train = regr.predict(n_x_train)
n_y_pred_test = regr.predict(n_x_test)

n_mse = mean_squared_error(n_y_test, n_y_pred_test)

plt.figure()
plt.title("Normalization")
plt.scatter(n_y_train, n_y_pred_train, color='r', s=10, label='train')
plt.scatter(n_y_test, n_y_pred_test, color='b', s=10, label='test')
plt.plot([-4, 4], [-1, 1])
plt.legend(fontsize='small')
plt.text(-4,0.7,"MSE : %.2f"%n_mse)

s_y = (-2*s_x) + 1 + (1.2*noise)

s_x_train,s_x_test,s_y_train,s_y_test = train_test_split(s_x, s_y, test_size=0.2)

regr = LinearRegression()
regr.fit(s_x_train, s_y_train)
s_y_pred_train = regr.predict(s_x_train)
s_y_pred_test = regr.predict(s_x_test)

s_mse = mean_squared_error(s_y_test, s_y_pred_test)

plt.figure()
plt.title("Standardization")
plt.scatter(s_y_train, s_y_pred_train, color='r', s=10, label='train')
plt.scatter(s_y_test, s_y_pred_test, color='b', s=10, label='test')
plt.plot([-6, 8], [-6, 8])
plt.legend(fontsize='small')
plt.text(-5,6,"MSE : %.2f"%s_mse)