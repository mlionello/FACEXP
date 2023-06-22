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
    pca = PCA(n_components =20)
    pca.fit(X[training_indices, :])
    X = pca.transform(X)

    y_test = y[test_indices]
    X_test = X[test_indices, :]
    y_training = y[training_indices]
    X_training = X[training_indices, :]

    model = knnc(n_neighbors=5)
    model.fit(X_training, y_training)
    score = model.score(X_training, y_training)
    print(f"for actor {actor_ind}, training score (len {np.sum(training_indices)}): {score:.3f};", end=' ')
    training_score.append(score)
    score = model.score(X_test, y_test)
    y_hat = model.predict(X_test)
    print(f"test score (len {np.sum(test_indices)}): {score:.3f};", end='\n')
    test_score.append(score)
    preds.append([y_hat,y_test])

print(f"training m: {np.mean(training_score):.3f}, std:  {np.std(training_score):.3f},"
      f"test m: {np.mean(test_score):.3f}, "
      f"max {np.max(test_score):.3f}  min {np.min(test_score):.3f}, "
      f"std:  {np.std(test_score):.3f}")


valid_indices = (mod == 2)
#valid_indices = valid_indices & intensity == 2
cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
model = knnc(n_neighbors=5)
scorestot = cross_validate(model, X[valid_indices, :], y[valid_indices], cv=cv, return_train_score=True)
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
