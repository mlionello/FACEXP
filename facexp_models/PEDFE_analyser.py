import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA


def get_sample_increase_models(X, y):
    score = []
    for k in range(50, X.shape[0], 50):
        score_k = get_cv_results(X[:k, :], y)
        score.append(score_k)
    plt.plot(pltscore[:, 0], label="test")
    plt.plot(pltscore[:, 1], label="train")

    plt.legend()
    plt.savefig("features_allsamples.png")

def get_hitrate_decrease_models(X, y, hr_scores):
    score = []
    for k in range(50, X.shape[0], 50):
        score_k = get_cv_results(X[:k, :], y)
        score.append(score_k)
    plt.plot(pltscore[:, 0], label="test")
    plt.plot(pltscore[:, 1], label="train")

    plt.legend()
    plt.savefig("features_allsamples.png")


def get_cv_results(X, y):
    confused_mat = []
    scores = []
    pca = PCA(n_components=20)
    model = knnc(n_neighbors=5)
    cv = StratifiedKFold(n_splits=20, random_state=0, shuffle=True)
    for train_ndx, test_ndx in cv.split(X, y):
        train_X, train_y, test_X, test_y = (
            X[train_ndx],
            y[train_ndx],
            X[test_ndx],
            y[test_ndx],
        )
        train_X = pca.fit_transform(train_X)
        test_X = pca.transform(test_X)
        model.fit(train_X, train_y)
        y_tr_pred = model.predict(train_X)
        y_tst_pred = model.predict(test_X)

        score_tr = np.sum(y_tr_pred == train_y) / len(train_y)
        score_tst = np.sum(y_tst_pred == test_y) / len(test_y)
        cm = confusion_matrix(test_y, y_tst_pred)

        scores.append([score_tr, score_tst])
        confused_mat.append(cm)

    scores = np.array(scores)
    confused_mat = np.array(confused_mat)
    mean_cm = np.mean(confused_mat, axis=0)
    mean_scores = np.mean(scores, axis=1)
    return {
        "scores": scores,
        "conf_matrices": confused_mat,
        "mean_cm": mean_cm,
        "mean_scores": mean_scores,
    }


features_path = "./features.npy"
labels_path = "./labels.npy"

X = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
hr_scores = np.array(labels[:, 5])
print(f"total number of subjects: {X.shape[0]}")

emo_id = labels[:, 4]
emo_id = [str(a) for a in emo_id]
y = np.reshape(emo_id, (-1,))

posed_indices = labels[:, 3] == "Posed"
Xp = X[posed_indices, :]
yp = y[posed_indices]
genuine_indices = labels[:, 3] == "Genuine"
Xg = X[genuine_indices, :]
yg = y[genuine_indices]

# Overall results
pca = PCA(n_components=20)
model = knnc(n_neighbors=5)
cv = StratifiedKFold(n_splits=20, random_state=0, shuffle=True)
Xp_pca = pca.fit_transform(Xp)
Xg_pca = pca.fit_transform(Xg)
Xall_pca = pca.fit_transform(X)
scoresp = cross_validate(model, Xp_pca, yp, cv=cv, return_train_score=True)
print(
    f"Posed: tr: {np.mean(scoresp['train_score']):.3f} +/- {np.std(scoresp['train_score']):.3f};"
    f"  tst: {np.mean(scoresp['test_score']):.3f} +/- {np.std(scoresp['test_score']):.3f}"
)
scoresg = cross_validate(model, Xg_pca, yg, cv=cv, return_train_score=True)
print(
    f"Genuine: tr: {np.mean(scoresg['train_score']):.3f} +/- {np.std(scoresg['train_score']):.3f};"
    f"  tst: {np.mean(scoresg['test_score']):.3f} +/- {np.std(scoresg['test_score']):.3f}"
)
scorestot = cross_validate(model, Xall_pca, y, cv=cv, return_train_score=True)
print(
    f"Whole: tr: {np.mean(scorestot['train_score']):.3f} +/- {np.std(scorestot['train_score']):.3f};"
    f"  tst: {np.mean(scorestot['test_score']):.3f} +/- {np.std(scorestot['test_score']):.3f}"
)

for x_data, y_data in [[Xp, yp], [Xg, yg], [X, y]]:
    get_sample_increase(x_data, y_data)
    get_hitrate_decrease(x_data, y_data, hr_scores)

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
