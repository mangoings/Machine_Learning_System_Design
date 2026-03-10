#rkdalsrb

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11, 1)

f_x1 = x**2
x1_new = 10
p = 0.1
derive = []
y = []

for i in range(10):
    x1_old = x1_new
    derive.append(x1_old - p*2*x1_old)
    x1_new = x1_old - p*2*x1_old
    y.append(x1_new**2)

plt.figure()
plt.scatter(derive, y)
plt.plot(x, f_x1)

xx = np.arange(-3, 3.01, 0.01)

f_x2 = xx*(np.sin(xx**2))
x2_new = 1.6
p = 0.01
derive2 = []
y2 = []

for i in range(10):
    x2_old = x2_new
    derive2.append(x2_old - p*(np.sin(x2_old**2) + x2_old*np.cos(x2_old**2)*2*x2_old))
    x2_new = x2_old - p*(np.sin(x2_old**2) + x2_old*np.cos(x2_old**2)*2*x2_old)
    y2.append(x2_new*np.sin(x2_new**2))

plt.figure()
plt.scatter(derive2, y2)
plt.plot(xx, f_x2)
