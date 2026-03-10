# 2019116278 강민규
import os

os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn import cluster

iris = load_iris()
X = iris.data  
y = iris.target  

model = cluster.KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)

accuracy = accuracy_score(y, labels)

print("iris 데이터의 군집화 정확도:", accuracy)
