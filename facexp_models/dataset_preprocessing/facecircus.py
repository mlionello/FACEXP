import h5py
import numpy as np
import scipy.io
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import h5py
import numpy as np
import scipy.io
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.decomposition import PCA


def preprocess_h5_files(datapath):
    # import metadata
    mat = scipy.io.loadmat(datapath / 'FaceRatings.mat')
    face_ratings = mat['FaceRatings']
    mat = scipy.io.loadmat(datapath / 'FaceRatingsMetaData.mat')
    FaceRatingsMetaData = mat['FaceRatingsMetaData']

    mat = scipy.io.loadmat(datapath / 'FaceRatingsOverlap.mat')
    FaceRatingsOverlap = mat['FaceRatingsOverlap']
    mat = scipy.io.loadmat(datapath / 'FaceRatingsProcessed.mat')
    FaceRatingsProcessed = mat['FaceRatingsProcessed']

    # extract subj_id code
    subj_id = FaceRatingsMetaData[0][0][0]
    new_subjid = [str(id[0][0]) for id in subj_id]

    # load vertices, filenames and shapes from h5
    file_ids = []
    meshes = []
    shapes = []
    in_folder = datapath / 'h5out' / 'local'

    file2load = Path('/home/matteo/Code/FACEXP/facexp_models/dataset_preprocessing/data_L2norm.npy')

    for j, fileh5 in enumerate(in_folder.glob("*.h5")):
        print(f"loading: {str(fileh5)}", end="\n")
        with h5py.File(fileh5) as data_in:
            file_ids.append(fileh5)

            meshes.append(np.array(data_in["v"]))
            shapes.append(data_in["v"].shape[0])

    # get minimum t len among subjects (should vary just in the order of tens of ms)
    min_t = np.min(shapes)
    # prepare subj baseline vertices array
    data = np.array([mesh[:min_t, :, :] for mesh in meshes])

    # fill neut array with 30-seconds-random-samples from each subject when rating is less than or eq to 25
    # neut_t_size = 30
    # data_neut = np.NAN*np.zeros((data.shape[0], neut_t_size, data.shape[2], data.shape[3]))
    # data_scaled = np.zeros(data.shape)
    # for j, subj_code in enumerate(new_subjid):
    #     rating_jth = FaceRatingsProcessed[0, :, j]
    #     t_neut_rating = np.where(rating_jth <= 25)[0]
    #     t_neut_rating_subset = np.random.choice(t_neut_rating, neut_t_size, replace=False)
    #     subjcode2vidindx = [j for j, file_id in enumerate(file_ids) if subj_code in str(file_id)]
    #     if len(subjcode2vidindx) == 0:
    #         print(f"{subj_code} not processed")
    #         continue
    #     data_neut[subjcode2vidindx, :, :, :] = data[subjcode2vidindx, t_neut_rating_subset*fps, :, :]

    # within subj displacement compared to mean of data_neut
    # data_scaled[subjcode2vidindx, :, :, :] = pdist(data[subjcode2vidindx, :, :, :]) - pdist(np.mean(data_neut[subjcode2vidindx, :, :, :]))

    # data_neut_mean = np.mean(data_neut, axis=1)
    # diff_data = data[:, :-1, :, :] - data[:, 1:, :, :]
    #
    # data_zscored_diff = data_zscored[:, :-1, :, :]-data_zscored[:, 1:, :, :]
    #
    # import matplotlib.pyplot as plt
    # plt.quiver(data_neut_mean[0, :, 0], data_neut_mean[0, :, 1], data_zscored_diff[0, 0, :, 1], data_zscored_diff[0, 0, :, 1])

    data_L2norm = np.sqrt(np.sum(data[:, :, :, [0, 1]] ** 2, axis=-1))  # only x and y axis
    direction_L2 = data / data_L2norm[..., np.newaxis]

    return data, data_L2norm, direction_L2


def compute_isc(data_L2norm, pca_component, fps=30, dopca=1):

    # compute zscore along time dimension
    data_L2_zscored = stats.zscore(data_L2norm, axis=1)
    nb_subj = data_L2norm.shape[0]
    timepoints = data_L2norm.shape[1]
    nb_features = data_L2norm.shape[2]

    w_lens = np.arange(2, 20, 2)*fps
    nb_windows = len(w_lens)

    indata = data_L2norm
    if dopca:
        nb_features = pca_component
        pca = PCA(n_components=pca_component)
        concat_L2_norm_zscored = np.reshape(data_L2_zscored,[-1, data_L2_zscored.shape[2]])
        pca.fit(concat_L2_norm_zscored)

        L2_norm_zscored_pca = np.zeros([data_L2_zscored.shape[0], data_L2_zscored.shape[1], pca_component])*np.nan
        L2_norm_zscored_pca[subj, :, :] = pca.transform(data_L2_zscored[subj])
        for subj in range(data_L2_zscored.shape[0]):
            L2_norm_zscored_pca[subj, :, :] = pca.transform(data_L2_zscored[subj])
        indata = L2_norm_zscored_pca

    isc = np.zeros([nb_windows+1, nb_subj, nb_subj, nb_features])*np.nan

    for win_j, t_win in enumerate(w_lens):
        print(f"\rt_window {win_j+1} / {len(w_lens)}", end='')
        hop_size = np.int32(t_win/3)
        t_range = np.arange(timepoints)
        t_range = [
            t_range[t * hop_size: t * hop_size + t_win]
            for t in range(np.int32(np.floor((timepoints - t_win) / hop_size)))
        ]
        for vx in range(nb_features):
            win_avg = np.mean(indata[:, t_range, vx],2)
            isc[win_j, :, :, vx] = np.corrcoef(win_avg)
            if win_j == nb_windows:
                isc[-1, :, :, vx] = np.corrcoef(indata[:, :, vx])

    return isc

if __name__=="__main__":
    dopca = 0
    pca_component = 20
    file2load = Path('/home/matteo/Code/FACEXP/facexp_models/dataset_preprocessing/data_L2norm.npy')
    if file2load.exists():
        data_L2norm = np.load(file2load)
    else:
        datapath = Path('/data1/EMOVIE_sampaolo/FACE/FaceCircus/data/')
        datapath = Path('/home/matteo/Code/FACEXP/data')
        data, data_L2norm, data_L2_direction = preprocess_h5_files(datapath)
    isc = compute_isc(data_L2norm, pca_component, dopca=dopca)
