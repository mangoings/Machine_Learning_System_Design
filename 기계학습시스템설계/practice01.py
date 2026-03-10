#강민규
import numpy as np

a = int(input("N을 입력하세요 >> "))
b = np.random.randint(1, 100, size=a)

print("arr : {}".format(b))

my_norm1 = float(sum(abs(b)))

def np_norm1(b):
    np_abs = np.abs(b)
    np_sum = np.sum(np_abs)
    return np_sum 

print("내가 만든 L1-Norm :",my_norm1)
print("numpy의 L1-Norm :",float(np_norm1(b)))

import math
my_norm2 = float(math.sqrt(sum(b**2)))

def np_norm2(b):
    np_multy = b*b
    np_sum2 = np.sum(np_multy)
    np_sqrt = np.sqrt(np_sum2)
    return np_sqrt

print("내가 만든 L2-Norm :",my_norm2)
print("numpy의 L2-Norm :",float(np_norm2(b)))


    
