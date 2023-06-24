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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def sample_size_cv(training_ind_range, X, y, training_indices, test_indices):
    smpl_size_score = []
    pca = PCA(n_components=20)
    pca.fit(X[training_indices, :])
    X = pca.transform(X)

    for k in range(len(training_ind_range)):
        same_fold_score = []
        for f in range(10):
            print(
                f"Processing {k + 1} fold out of {len(training_ind_range)}... {f + 1}",
                end="\r",
            )
            perm_training_ind = np.random.permutation(training_indices)
            tr_ind = perm_training_ind[: training_ind_range[k]]
            model = knnc(n_neighbors=5)
            model.fit(X[tr_ind, :], y[tr_ind])

            training_score = model.score(X[tr_ind], y[tr_ind])
            test_score = model.score(X[test_indices], y[test_indices])
            same_fold_score.append([training_score, test_score])
            if k == len(training_ind_range) - 1:
                break
        smpl_size_score.append(
            [np.mean(same_fold_score, 0), np.std(same_fold_score, 0)]
        )
    return np.array(smpl_size_score)


features_path = "./features.npy"
labels_path = "./labels.npy"

X = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
hit_rate = np.where(labels[:, 5] > 70)[0]
X = X[hit_rate, :]
labels = labels[hit_rate, :]

emo_id = labels[:, 4]
emo_id = [str(a) for a in emo_id]
y = np.reshape(emo_id, (-1,))

tr_score = []
tst_score = []

for i in range(100):
    print(f"cv folding repetition {i}")
    # cv = StratifiedKFold(n_splits=15, random_state=0, shuffle=True)
    # partitions = cv.split(X, y)
    # indices_first_part = next(partitions)
    # training_indices = indices_first_part[0]
    # test_indices = indices_first_part[1]
    emhist = [np.sum(y == em) for em in np.unique(y)]
    min_em_counter = np.min(emhist)
    equal_ind = [
        np.random.choice(np.where(y == em)[0], min_em_counter) for em in np.unique(y)
    ]
    equal_ind = np.reshape(equal_ind, (-1,))
    yeq = y[equal_ind]
    Xeq = X[equal_ind, :]
    test_indices = []
    for em in np.unique(y):
        all_occurancies = np.where(yeq == em)[0]
        test_indices.append(np.random.choice(all_occurancies, 1, replace=False))
    test_indices = np.reshape(test_indices, (-1,))
    training_indices = [ind for ind in range(len(yeq)) if ind not in test_indices]

    training_ind_range = np.arange(len(test_indices), len(training_indices), 40)
    scores_cv = sample_size_cv(
        training_ind_range, Xeq, yeq, training_indices, test_indices
    )
    tr_score.append(scores_cv[:, 0, 0])
    tst_score.append(scores_cv[:, 0, 1])

plt.plot(np.mean(tr_score, 0), label="training_score")
plt.plot(np.mean(tst_score, 0), label="test_score")
plt.legend()
plt.savefig("smpl_size_score.png")
np.save("smpl_size_score_1fold", np.array([tr_score, tst_score]))
