import argparse
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.decomposition import PCA


def run_analyse(pathfolder, outpathfolder, custom_cond, pca_n=20, k_nn=5):
    features_path = pathfolder / "features.npy"
    labels_path = pathfolder / "labels.npy"

    X = np.load(features_path, allow_pickle=True)
    labels_init = np.load(labels_path, allow_pickle=True)

    print(f"total number of subjects: {X.shape[0]}")
    print(custom_cond)

    labels_ids = labels_init[()]["column_names"]
    labels = labels_init[()]["labels"]
    labels = np.array(labels)
    em_col_ind = np.where([a == "Emotion" for a in labels_ids])[0][0]
    actor_col_ind = np.where([a == "Actor" for a in labels_ids])[0][0]
    rep_col_ind = np.where([a == "Repetition" for a in labels_ids])[0][0]
    ch_col_ind = np.where([a == "Channel" for a in labels_ids])[0][0]
    int_col_ind = np.where([a == "Intensity" for a in labels_ids])[0][0]
    sttm_col_ind = np.where([a == "Statement" for a in labels_ids])[0][0]
    mod_col_ind = np.where([a == "Modality" for a in labels_ids])[0][0]

    # N.B.
    emotions = labels[:, em_col_ind]
    actors = labels[:, actor_col_ind]
    rep = labels[:, rep_col_ind]
    ch = labels[:, ch_col_ind]
    intensity = labels[:, int_col_ind]
    sttm = labels[:, sttm_col_ind]
    mod = labels[:, mod_col_ind]

    valid_indices = mod == 2
    valid_indices = valid_indices & (emotions > 2)

    tr_custom_ind = [True] * labels.shape[0]
    tst_custom_ind = [True] * labels.shape[0]
    if custom_cond["tr_intensity"] > 0:
        tr_custom_ind = tr_custom_ind & (intensity == custom_cond["tr_intensity"])
    if custom_cond["tr_ch"] > 0:
        tr_custom_ind = tr_custom_ind & (ch == custom_cond["tr_ch"])
    if custom_cond["tr_rep"] > 0:
        tr_custom_ind = tr_custom_ind & (rep == custom_cond["tr_rep"])
    if custom_cond["tst_intensity"] > 0:
        tst_custom_ind = tst_custom_ind & (intensity == custom_cond["tst_intensity"])
    if custom_cond["tst_ch"] > 0:
        tst_custom_ind = tst_custom_ind & (ch == custom_cond["tst_ch"])
    if custom_cond["tst_rep"] > 0:
        tst_custom_ind = tst_custom_ind & (rep == custom_cond["tst_rep"])

    # if channel is not only speech: remove actor 18th who does not sing!
    if custom_cond["tr_ch"] != 1:
        tr_custom_ind = tr_custom_ind & (actors != 18)
    if custom_cond["tst_ch"] != 1:
        tst_custom_ind = tst_custom_ind & (actors != 18)

    y = np.reshape(emotions, (-1,))
    unique_actors = np.unique(actors)
    training_score = []
    test_score = []
    preds = []

    pca = PCA(n_components=pca_n)
    X0 = pca.fit_transform(X[valid_indices, :])
    model = knnc(n_neighbors=k_nn)
    cv = StratifiedKFold(10)
    scores = cross_validate(model,X0, y[valid_indices], cv=cv, return_train_score=True)
    print(np.mean(scores["test_score"]))
    print(np.mean(scores["train_score"]))

    unique_classes = np.unique(y)
    print(unique_classes)
    print(np.unique(scores["test_score"]))

    for cls in unique_classes:
        class_indices = np.where(y[valid_indices] == cls)[0]
        class_test_scores = scores["test_score"][class_indices]
        print(f"Class {cls}:")
        print(f"  Mean test score: {np.mean(class_test_scores)}")
    return

    for actor_ind in unique_actors:
        # if actor_ind == 18:
        #     continue
        actor_target_indices = actors == actor_ind
        training_indices = ~actor_target_indices & valid_indices & tr_custom_ind
        test_indices = actor_target_indices & valid_indices & tst_custom_ind

        pca = PCA(n_components=pca_n)
        pca.fit(X[training_indices, :])
        X_pca = pca.transform(X)

        y_test = y[test_indices]
        X_test = X_pca[test_indices, :]
        y_training = y[training_indices]
        X_training = X_pca[training_indices, :]

        model = knnc(n_neighbors=k_nn)
        model.fit(X_training, y_training)
        score = model.score(X_training, y_training)
        print(
            f"for actor {actor_ind}, training score (len {np.sum(training_indices)}): {score:.3f};",
            end=" ",
        )
        training_score.append(score)
        score = model.score(X_test, y_test)
        y_hat = model.predict(X_test)
        print(f"test score (len {np.sum(test_indices)}): {score:.3f};", end="\n")
        test_score.append(score)
        preds.append([y_hat, y_test])

    print(
        f"training m: {np.mean(training_score):.3f}, std:  {np.std(training_score):.3f},"
        f"test m: {np.mean(test_score):.3f}, "
        f"max {np.max(test_score):.3f}  min {np.min(test_score):.3f}, "
        f"std:  {np.std(test_score):.3f}"
    )

    cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    model = knnc(n_neighbors=k_nn)
    pca = PCA(n_components=pca_n)
    X_pca_valid = pca.fit_transform(X[valid_indices, :])
    scorestot = cross_validate(
        model, X_pca_valid, y[valid_indices], cv=cv, return_train_score=True
    )
    print(
        f"Whole: tr: {np.mean(scorestot['train_score']):.3f} +/- {np.std(scorestot['train_score']):.3f};"
        f"  tst: {np.mean(scorestot['test_score'])} +/- {np.std(scorestot['test_score'])}"
    )

    scores = {
        "training_score": training_score,
        "test_score": test_score,
        "preds": preds,
        "scorestot": scorestot,
        "k": k_nn,
        "pca_n": pca_n,
    }
    scores.update(custom_cond)

    if not outpathfolder.is_dir():
        os.mkdir(outpathfolder)
    with open(outpathfolder / "data.pkl", "wb") as tofile:
        pickle.dump(scores, tofile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run analysis on RAVDESS dataset")
    parser.add_argument(
        "--input", type=str, help="input folder to features.npy and labels.npy"
    )
    parser.add_argument("--output", type=str, default="./", help="output folder")
    parser.add_argument(
        "--tr_intensity",
        type=int,
        default="0",
        help="intensity level in training set (1 weak 2 strong)",
    )
    parser.add_argument(
        "--tr_ch",
        type=int,
        default="0",
        help="channel number in training set (1 speech, 2 singing)",
    )
    parser.add_argument(
        "--tr_rep", type=int, default="0", help="number of repetitions in training set"
    )
    parser.add_argument(
        "--tst_intensity",
        type=int,
        default="0",
        help="channel number in testing set (1 speech, 2 singing)",
    )
    parser.add_argument(
        "--tst_ch",
        type=int,
        default="1",
        help="channel number in testing set (1 speech, 2 singing)",
    )
    parser.add_argument("--tst_rep", type=int, default="0", help="output folder")
    args = parser.parse_args()
    custom_cond = {
        "tr_intensity": args.tr_intensity,
        "tr_ch": args.tr_ch,
        "tr_rep": args.tr_rep,
        "tst_intensity": args.tst_intensity,
        "tst_ch": args.tst_ch,
        "tst_rep": args.tst_rep,
    }
    if custom_cond["tst_ch"] != 1:
        raise Exception(
            "Sorry, test set cannot have singing samples as not all emotions are covered"
        )
    if custom_cond["tr_ch"] == 2:
        raise Exception(
            "Sorry, training set cannot have only singing samples, which do not cover all the emotions"
        )

    run_analyse(Path(args.input), Path(args.output), custom_cond)
