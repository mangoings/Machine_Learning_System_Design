import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(500)
noise = np.random.randn(len(x))
y = (-2*x) + 1 + (1.2*noise)

x_reshaped = x.reshape(-1,1)
X = np.c_[np.ones((500, 1)), x_reshaped]

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def h(x, theta):
    return x*theta[1] + theta[0]

w = theta[1]
b = theta[0]

plt.figure()
plt.scatter(x,y, s=10)
plt.plot([-3, 4], [h(-3, theta), h(4, theta)], 'r')
plt.xlabel('X - input')
plt.ylabel('y - target / True')
plt.text(1, 6, "W = [%f]"%w)
plt.text(1, 5, "b = [%f]"%b)