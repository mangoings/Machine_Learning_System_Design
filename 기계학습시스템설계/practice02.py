import numpy as np

a = int(input("행렬의 k를 입력하세요 >> "))

b = np.random.randint(1, 10, size=[a, a])
c = np.random.randint(1, 10, size=[a, a])

print("행렬 A\n {}".format(b))
print("행렬 B\n {}".format(c))

def my_hadamard(x, y):
    h = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
        for j in range(len(y)):
            h[i][j] = x[i][j] * y[i][j]
    return h

print("내가 만든 hadamard product\n {}".format(my_hadamard(b, c)))
print("numpy의 hadamard product\n {}".format(np.multiply(b, c)))

n, m = map(int,input("행렬의 n, m을 입력하세요 >> ").split())

d = np.random.randint(1, 10, size=[n, m])
e = np.random.randint(1, 10, size=[m, n])

print("행렬 A\n {}".format(d))
print("행렬 B\n {}".format(e))

def my_dot(x, y):
    g = np.zeros((x.shape[0], y.shape[1]))
    for i in range(len(x)):
        for j in range(x.shape[1]):
            for k in range(y.shape[1]):
                g[i][k] += x[i][j] * y[j][k]
    return g

print("내가 만든 dot product\n {}".format(my_dot(d, e)))
print("numpy의 dot product\n {}".format(np.matmul(d, e)))



    