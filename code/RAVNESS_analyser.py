import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier as knnc

from sklearn.decomposition import PCA

pca_n = 20
k_nn = 5

features_path = "../features.npy"
labels_path = "../labels.npy"

X = np.load(features_path, allow_pickle=True)
labels_init = np.load(labels_path, allow_pickle=True)

print(f"total number of subjects: {X.shape[0]}")

labels_ids = labels_init[()]["column_names"]
labels = labels_init[()]['labels']
labels = np.array(labels)
em_col_ind = np.where([a == "Emotion" for a in labels_ids])[0][0]
actor_col_ind = np.where([a == "Actor" for a in labels_ids])[0][0]
rep_col_ind = np.where([a == "Repetition" for a in labels_ids])[0][0]
ch_col_ind = np.where([a == "Channel" for a in labels_ids])[0][0]
int_col_ind = np.where([a == "Intensity" for a in labels_ids])[0][0]
sttm_col_ind = np.where([a == "Statement" for a in labels_ids])[0][0]
mod_col_ind = np.where([a == "Modality" for a in labels_ids])[0][0]

emotions = labels[:, em_col_ind]
actors = labels[:, actor_col_ind]
rep = labels[:, rep_col_ind]
ch = labels[:, ch_col_ind]
intensity = labels[:, int_col_ind]
sttm = labels[:, sttm_col_ind]
mod = labels[:, mod_col_ind]

y = np.reshape(emotions, (-1,))

valid_indices = (mod == 2) & (actors != 18)
valid_indices = valid_indices & (emotions > 2)
valid_indices = valid_indices & (rep == 2)
#valid_indices = valid_indices & (intensity == 2)

unique_actors = np.unique(actors)
training_score = []
test_score = []
preds = []
for actor_ind in unique_actors:
    if actor_ind == 18:
        continue
    actor_target_indices = (actors == actor_ind)
    training_indices = ~actor_target_indices & valid_indices
    test_indices = actor_target_indices & valid_indices
    test_indices = test_indices & (ch == 2) & (intensity == 2)
    pca = PCA(n_components =pca_n)
    pca.fit(X[training_indices, :])
    X_pca = pca.transform(X)

    y_test = y[test_indices]
    X_test = X_pca[test_indices, :]
    y_training = y[training_indices]
    X_training = X_pca[training_indices, :]

    model = knnc(n_neighbors=k_nn)
    model.fit(X_training, y_training)
    score = model.score(X_training, y_training)
    print(f"for actor {actor_ind}, training score (len {np.sum(training_indices)}): {score:.3f};", end=' ')
    training_score.append(score)
    score = model.score(X_test, y_test)
    y_hat = model.predict(X_test)
    print(f"test score (len {np.sum(test_indices)}): {score:.3f};", end='\n')
    test_score.append(score)
    preds.append([y_hat, y_test])

print(f"training m: {np.mean(training_score):.3f}, std:  {np.std(training_score):.3f},"
      f"test m: {np.mean(test_score):.3f}, "
      f"max {np.max(test_score):.3f}  min {np.min(test_score):.3f}, "
      f"std:  {np.std(test_score):.3f}")


cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
model = knnc(n_neighbors=k_nn)
pca = PCA(n_components=pca_n)
X_pca_valid = pca.fit_transform(X[valid_indices, :])
scorestot = cross_validate(model, X_pca_valid, y[valid_indices], cv=cv, return_train_score=True)
print(f"Whole: tr: {np.mean(scorestot['train_score']):.3f} +/- {np.std(scorestot['train_score']):.3f};"
      f"  tst: {np.mean(scorestot['test_score'])} +/- {np.std(scorestot['test_score'])}")

scores = {
    "training_score": training_score,
    "test_score": test_score,
    "preds": preds,
    "scorestot": scorestot,
    "k": k_nn,
    "pca_n" : pca_n,
}

with open("data.pkl", "wb") as tofile:
    pickle.dump(scores, tofile)
