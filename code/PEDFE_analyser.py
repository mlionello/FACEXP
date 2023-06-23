import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

features_path = "./features.npy"
labels_path = "./labels.npy"

X = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
hit_rate = np.where(labels[:, 5] > 70)[0]
X = X[hit_rate, :]
labels = labels[hit_rate, :]
print(f"total number of subjects: {X.shape[0]}")

pca = PCA(n_components=20)
X = pca.fit_transform(X)

posed_indices = np.where(labels[:, 3] == 'Posed')[0]
Xp = X[posed_indices, :]
yp = labels[posed_indices, 4]
genuine_indices = np.where(labels[:, 3] == 'Genuine')[0]
Xg = X[genuine_indices, :]
yg = labels[genuine_indices, 4]

emo_id = labels[:, 4]
emo_id = [str(a) for a in emo_id]
y = np.reshape(emo_id, (-1,))

cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
model = knnc(n_neighbors=3)



scoresp = cross_validate(model, Xp, yp, cv=cv, return_train_score=True)
print(f"Posed: tr: {np.mean(scoresp['train_score']):.3f} +/- {np.std(scoresp['train_score']):.3f};"
      f"  tst: {np.mean(scoresp['test_score'])} +/- {np.std(scoresp['test_score'])}")
scoresg = cross_validate(model, Xg, yg, cv=cv, return_train_score=True)
print(f"Genuine: tr: {np.mean(scoresg['train_score']):.3f} +/- {np.std(scoresg['train_score']):.3f};"
      f"  tst: {np.mean(scoresg['test_score'])} +/- {np.std(scoresg['test_score'])}")
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
