import argparse
import os
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.decomposition import PCA

def run_analyse(infold, outfold, pca_n=20, k_nn = 1):
    X = np.load(infold / 'features_pilot.npy')
    labels = np.load(infold / 'labels_pilot.npy')
    indxremov = (labels[:,1]==1) | (labels[:,1]==2) | (labels[:,1]==6) | (labels[:,1]==3)
    labels = labels[~indxremov,:]
    X = X[~indxremov,:]

    y = labels[:, 1]
    actors = labels[:, 0]
    repetitions = labels[:, 2]
    if not outfold.is_dir():
        os.mkdir(outfold)

    # overall model
    pca = PCA(n_components=pca_n)
    X_pca = pca.fit_transform(X)  # these need to be moved inside the for loop

    scores = []
    for k_tmp in range(1, 6):
        scores_row = []
        for rep in list(np.unique(repetitions)):
            tr_repindices = repetitions <= rep
            X_i = X_pca[tr_repindices, :]
            y_i = y[tr_repindices]
            cv = StratifiedKFold(5)
            model = knnc(n_neighbors=k_tmp)
            score_tmp = cross_validate(model, X_i, y_i, cv=cv, return_train_score=True)
            scores_row.append([np.mean(score_tmp["train_score"]), np.mean(score_tmp["test_score"])])
        scores.append(scores_row)
    scores = np.array(scores)
    np.save(outfold/f'overall_scores_k1to5_cv5', scores)

    final_scores = []
    final_cm = []
    for strategy in [1, 2]:
        strategy_scores = []
        strategy_cm = []
        for rep in list(np.unique(repetitions)):
            print(f"modelling for {rep} repetitinos seen in the training set")
            tr_repindices = repetitions <= rep
            scores_across_actors = []
            cm_across_actors = []
            for actor_ind in list(np.unique(actors)):
                print(f"modelling for testing on actor n. {actor_ind}", end='\r')
                if strategy==1:
                    actor_target_indices = (actors == actor_ind) & tr_repindices
                    training_indices = ~actor_target_indices
                    test_indices = actor_target_indices
                else:
                    actor_target_indices = (actors == actor_ind)
                    training_indices = ~actor_target_indices & tr_repindices
                    test_indices = actor_target_indices

                pca = PCA(n_components=pca_n)
                pca.fit(X[training_indices, :])
                X_pca = pca.transform(X)

                y_test = y[test_indices]
                X_test = X_pca[test_indices, :]
                y_training = y[training_indices]
                X_training = X_pca[training_indices, :]

                model = knnc(n_neighbors=k_nn)
                model.fit(X_training, y_training)
                score_tr = model.score(X_training, y_training)
                score_tst = model.score(X_test, y_test)
                y_pred = model.predict(X_test)

                scores_across_actors.append([score_tr, score_tst])
                cm_across_actors.append(confusion_matrix(y_test, y_pred))
            strategy_scores.append(scores_across_actors)
            strategy_cm.append(cm_across_actors)
        final_scores.append(strategy_scores)
        final_cm.append(strategy_cm)
    final_cm = np.array(final_cm)
    final_scores = np.array(final_scores)
    np.save(outfold / 'scores_pilot', final_scores)
    np.save(outfold / 'cm_pilot', final_cm)
    plt.figure()
    [plt.plot(final_scores[0, :, j, 1], label=f'subj_{j + 1}') for j in range(6)]
    plt.legend()
    plt.savefig(outfold / 'repfromtsttraining')
    plt.show()
    plt.close()
    plt.figure()
    [plt.plot(final_scores[1, :, j, 1], label=f'subj_{j + 1}') for j in range(6)]
    plt.legend()
    plt.savefig(outfold / 'incr_repts_in_training')
    plt.show()
    plt.close()
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run analysis on RAVDESS dataset")
    parser.add_argument(
        "--input", default='../mediapipe/pilot/', type=str, help="input folder to features.npy and labels.npy"
    )
    parser.add_argument("--output", type=str, default="./results/pilot_results/", help="output folder")
    args = parser.parse_args()

    run_analyse(Path(args.input), Path(args.output))
