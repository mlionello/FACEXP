import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

features_path = "../features.npy"
labels_path = "../labels.npy"

X = np.load(features_path, allow_pickle=True)
labels_init = np.load(labels_path, allow_pickle=True)

# hit_rate = np.where(labels[:, 5] > 70)[0]
# X = X[hit_rate, :]
# labels = labels[hit_rate, :]

print(f"total number of subjects: {X.shape[0]}")

pca = PCA(n_components=20)
X = pca.fit_transform(X)

labels_ids = labels_init[()]["column_names"]
labels = labels_init[()]['labels']
labels = np.array(labels)
em_col_ind = np.where([a == "Emotion" for a in labels_ids])[0][0]
actor_col_ind = np.where([a == "Actor" for a in labels_ids])[0][0]
emotions = labels[:, em_col_ind]
actors = labels[:, actor_col_ind]

unique_actors = np.unique(actors)
print(unique_actors)
import sys
sys.exit()
for k in range(10):


y = np.reshape(emotions, (-1,))

cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
model = knnc(n_neighbors=3)

scorestot = cross_validate(model, X, y, cv=cv, return_train_score=True)
print(f"Whole: tr: {np.mean(scorestot['train_score']):.3f} +/- {np.std(scorestot['train_score']):.3f};"
      f"  tst: {np.mean(scorestot['test_score'])} +/- {np.std(scorestot['test_score'])}")

pltscore = []
for k in range(5, X.shape[1]):
    cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    scores = cross_validate(model, X[:, : k], y, cv=cv, return_train_score=True)
    pltscore.append([np.mean(scores["test_score"]), np.mean(scores["train_score"])])
pltscore = np.array(pltscore)
plt.plot(pltscore[:, 0], label='test')
plt.plot(pltscore[:, 1], label='train')

plt.legend()
plt.savefig('features_allsamples.png')

# predicted_classes = []
# actual_classes = []
# matrix = 0
# for train_ndx, test_ndx in cv.split(X, y):
#     model = knnc(n_neighbors=5)
#     train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
#
#     model.fit(train_X, train_y)
#     predicted_classes = model.predict(test_X)
#     cm = confusion_matrix(test_y, predicted_classes, labels=model.classes_)
#     matrix = matrix + cm
#     print(cm)
#     print(np.sum(np.diag(cm))/np.sum(cm))
#
# print(matrix)
# print(np.sum(np.diag(matrix))/np.sum(matrix))
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
# disp.plot()
# plt.show()
