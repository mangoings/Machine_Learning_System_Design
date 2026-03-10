# 2019116278 강민규

# (1)
import matplotlib.pyplot as plt
import numpy as np

dach_length = [75, 77, 83, 81, 73, 99, 72, 83]
dach_height = [24, 29, 19, 32, 21, 22, 19, 34]

samo_length = [76, 78, 82, 88, 76, 83, 81, 89]
samo_height = [55, 58, 53, 54, 61, 52, 57, 64]

malt_length = [35, 39, 38, 41, 30, 57, 41, 35]
malt_height = [23, 26, 19, 30, 21, 24, 28, 20]

d_data = np.column_stack((dach_length, dach_height))
d_label = np.zeros(len(d_data))
s_data = np.column_stack((samo_length, samo_height))
s_label = np.ones(len(s_data))
m_data = np.column_stack((malt_length, malt_height))
m_label = np.full(len(m_data), 2)

nd_data = d_data.tolist()
ns_data = s_data.tolist()
nm_data = m_data.tolist()

print("닥스훈트(0) :", nd_data)
print("사모예드(1) :", ns_data)
print("말티즈(2) :", nm_data)

# (2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

dogs = np.concatenate((d_data, s_data, m_data))
labels = np.concatenate((d_label, s_label, m_label))

dog_classes = {0: 'Dachshund', 1: 'Samoyed', 2: 'maltese'}

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs, labels)
prediction = knn.predict(dogs)

conf_mat = confusion_matrix(labels, prediction)
print("\n",conf_mat)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(conf_mat, cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(122)
plt.imshow(conf_mat, cmap=plt.cm.gray)
plt.colorbar()
ConfusionMatrixDisplay(confusion_matrix=conf_mat).plot()

# (3)
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def my_precision_score(y_true, y_pred, pos_label=2):
    true_positive = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    predicted_positive = (y_pred == pos_label).sum()
    return true_positive / predicted_positive

def my_recall_score(y_true, y_pred, pos_label=2):
    true_positive = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    actual_positive = (y_true == pos_label).sum()
    return true_positive / actual_positive

def my_accuracy_score(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    return correct / total

def my_f1_score(y_true, y_pred, pos_label=2):
    precision = my_precision_score(y_true, y_pred, pos_label)
    recall = my_recall_score(y_true, y_pred, pos_label)
    return 2 * (precision * recall) / (precision + recall)

prediction_dach = prediction[:len(dach_length)]
prediction_malt = prediction[-len(malt_length):]

labels_new = np.concatenate((d_label, m_label))
prediction_new = np.concatenate((prediction_dach, prediction_malt))

my_precision = my_precision_score(labels_new, prediction_new, pos_label=2)
my_recall = my_recall_score(labels_new, prediction_new, pos_label=2)
my_accuracy = my_accuracy_score(labels_new, prediction_new)
my_f1 = my_f1_score(labels_new, prediction_new, pos_label=2)

print("\n내가 만든 Precision:", my_precision)
print("내가 만든 Recall:", my_recall)
print("내가 만든 Accuracy:", my_accuracy)
print("내가 만든 F1-score:", my_f1)

sk_precision = precision_score(labels_new, prediction_new, pos_label=2)
sk_recall = recall_score(labels_new, prediction_new, pos_label=2)
sk_accuracy = accuracy_score(labels_new, prediction_new)
sk_f1 = f1_score(labels_new, prediction_new, pos_label=2)

print("\nscikit-learn의 Precision:", sk_precision)
print("scikit-learn의 Recall:", sk_recall)
print("scikit-learn의 Accuracy:", sk_accuracy)
print("scikit-learn의 F1-score:", sk_f1)

# (4)
new_data = np.array([
    [58, 30],
    [80, 26],
    [80, 41],
    [75, 55]
])

dog_classes_new = {0: '닥스훈트', 1: '사모예드', 2: '말티즈'}

for i, data_point in enumerate(new_data, start=65):
    print(f"\n{chr(i)} 분류 결과:")
    for k_value in [3, 5, 7]:
        knn_new = KNeighborsClassifier(n_neighbors=k_value)
        knn_new.fit(dogs, labels)
        prediction_new_data = knn_new.predict([data_point])
        breed = dog_classes_new[int(prediction_new_data[0])]
        print(f"{chr(i)} [{data_point}] : k={k_value} 일때 : {breed}")

# (5)
plt.figure()
plt.title('Dog size')
plt.xlabel('Length')
plt.ylabel('Height')
plt.scatter(dach_length, dach_height, c='red', marker='8', label='Dachshund')
plt.scatter(samo_length, samo_height, c='blue', marker='^', label='Samoyed')
plt.scatter(malt_length, malt_height, c='green', marker='s', label='maltese')
plt.scatter(new_data[0, 0], new_data[0, 1], c='fuchsia', marker='o', s=300, label='A')
plt.scatter(new_data[1, 0], new_data[1, 1], c='gray', marker='o', s=300, label='B')
plt.scatter(new_data[2, 0], new_data[2, 1], c='aqua', marker='o', s=300, label='C')
plt.scatter(new_data[3, 0], new_data[3, 1], c='green', marker='o', s=300, label='D')
plt.legend()

# (6)
from sklearn import cluster
import os

os.environ["OMP_NUM_THREADS"] = "1"

dog_data = np.concatenate((dogs, new_data))

dog_length = dog_data[:,0]
dog_height = dog_data[:,1]

plt.figure()
plt.title('Dog data without label')
plt.xlabel('Length')
plt.ylabel('Height')
plt.scatter(dog_length, dog_height, c='blue', marker='8', label='no labeled data')
plt.legend()

def kmeans_predict_plt(X, k):
    model = cluster.KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    labels = model.predict(X)
    colors = np.array(['red', 'green', 'blue', 'magenta'])
    plt.figure()
    plt.title('k-Means clustering, k-{}'.format(k))
    plt.scatter(X[:, 0], X[:, 1], color=colors[labels])
    
kmeans_predict_plt(dog_data, k=2)
kmeans_predict_plt(dog_data, k=3)
kmeans_predict_plt(dog_data, k=4)
