import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

dog_classes = {0:'Dachshund', 1:'Samoyed', 2:'Maltese'}

d_d_data = d_data.tolist()
s_s_data = s_data.tolist()
m_m_data = m_data.tolist()

print("닥스훈트(0) : {}".format(d_d_data))
print("사모예드(1) : {}".format(s_s_data))
print("말티즈(2) : {}".format(m_m_data))

dogs = np.concatenate((d_data, s_data, m_data))
labels = np.concatenate((d_label, s_label, m_label))

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs, labels)
y_pred = knn.predict(dogs)

conf_mat = confusion_matrix(labels, y_pred)

print("\n",conf_mat)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(conf_mat, cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(122)
plt.imshow(conf_mat, cmap=plt.cm.gray)
plt.colorbar()
ConfusionMatrixDisplay(confusion_matrix=conf_mat).plot()

y_pred_dach = y_pred[:len(dach_length)]
y_pred_malt = y_pred[-len(malt_length):]

new_y_pred = np.concatenate((y_pred_dach, y_pred_malt))
new_labels = np.concatenate((d_label, m_label))

def precision(y_true, y_pred, pos_label = 2):
    real_pred_dog = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    pred_dog = (y_pred == pos_label).sum()
    pre_dog = real_pred_dog / pred_dog
    return pre_dog

def recall(y_true, y_pred, pos_label = 2):
    real_dog = (y_true == pos_label).sum()
    real_pred_dog = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    rec_dog = real_pred_dog / real_dog
    return rec_dog

def accuracy(y_true, y_pred):
    acc_dog = (y_true == y_pred).sum() / len(y_true)
    return acc_dog

def F1_score(y_true, y_pred, pos_label = 2):
    dog_pre = precision(y_true, y_pred, pos_label = 2)
    dog_rec = recall(y_true, y_pred, pos_label = 2)
    F1_dog = 2*(dog_pre*dog_rec) / (dog_pre + dog_rec)
    return F1_dog

print("내가 만든 Precision: ", precision(new_labels, new_y_pred, pos_label = 2))
print("내가 만든 Recall: ", recall(new_labels, new_y_pred, pos_label = 2))
print("내가 만든 Accuracy: ", accuracy(new_labels, new_y_pred))
print("내가 만든 F1-score: ", F1_score(new_labels, new_y_pred, pos_label = 2))

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

print("scikit-learn의 Precision: ", precision_score(new_labels, new_y_pred, pos_label = 2))
print("scikit-learn의 Recall: ", recall_score(new_labels, new_y_pred, pos_label = 2))
print("scikit-learn의 Accuracy: ", accuracy_score(new_labels, new_y_pred))
print("scikit-learn의 F1-score: ", f1_score(new_labels, new_y_pred, pos_label = 2))

new_data = np.array([[58, 30],
                     [80, 26],
                     [80, 41],
                     [75, 55]])

new_classes = {0:'닥스훈트', 1:'사모예드', 2:'말티즈'}

j = 0

for i in ['A', 'B', 'C', 'D']:
    print("\n%s 데이터 분류결과"%i)
    for k_new in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k_new)
        knn.fit(dogs, labels)
        y = knn.predict(new_data)
        t = int(y[j])
        print("{:s} {} : k가 {:d} 일때 : {:s}".format(i,new_data[j],k_new,new_classes[t]))
    j += 1

plt.figure()
plt.scatter(dach_length, dach_height, color='r', marker='8', label='Dachshund')
plt.scatter(samo_length, samo_height, color='b', marker='^', label='Samoyed')
plt.scatter(malt_length, malt_height, color='g', marker='s', label='Maltese')
plt.scatter(new_data[0, 0], new_data[0, 1], color='pink', marker='o', s=300, label='A')
plt.scatter(new_data[1, 0], new_data[1, 1], color='gray', marker='o', s=300, label='B')
plt.scatter(new_data[2, 0], new_data[2, 1], color='aqua', marker='o', s=300, label='C')
plt.scatter(new_data[3, 0], new_data[3, 1], color='green', marker='o', s=300, label='D')
plt.legend()

dog_data = np.concatenate((dogs, new_data))

X = dog_data[:,0]
Y = dog_data[:,1]

plt.figure()
plt.title("Dog data without label")
plt.xlabel("Length")
plt.ylabel("Height")
plt.scatter(X, Y, color='b', marker='8', label='no labeled data')
plt.legend()

from sklearn import cluster

def kmeans_predict_plot(x, k):
    model = cluster.KMeans(n_clusters=k)
    model.fit(x)
    labels = model.predict(x)
    colors = np.array(['red', 'green', 'blue', 'magenta'])
    plt.title('k-Means clustering, k=%d'%k)
    plt.scatter(x[:,0], x[:,1], color=colors[labels])

plt.figure()
kmeans_predict_plot(dog_data, 2)
plt.figure()
kmeans_predict_plot(dog_data, 3)
plt.figure()
kmeans_predict_plot(dog_data, 4)