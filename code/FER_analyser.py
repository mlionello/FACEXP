import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

features_path = "./features.npy"
labels_path = "./labels.npy"

features = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
X = features.transpose()
# hit_rate = np.where(labels[:, 5]>50)[0]
# X = X[hit_rate, :]
# labels = labels[hit_rate, :]

emo_id = labels[:, 4]
emo_id = [str(a) for a in emo_id]
y = np.reshape(emo_id, (-1, 1))

cv = KFold(n_splits=5, random_state=1, shuffle=True)

predicted_classes = []
actual_classes = []
matrix = 0
for train_ndx, test_ndx in cv.split(X):
    model = knnc(n_neighbors=5)
    train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

    model.fit(train_X, train_y)
    predicted_classes = model.predict(test_X)
    cm = confusion_matrix( test_y , predicted_classes, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()
    matrix = matrix + cm
    print(cm)
    print(np.sum(np.diag(cm))/np.sum(cm))

print(matrix)
print(np.sum(np.diag(matrix))/np.sum(matrix))
