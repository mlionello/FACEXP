import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def get_mv_avg(data, win=1):
    series = pd.Series(data)
    moving_average = series.rolling(window=win).mean()
    return moving_average

with open('../output_posed_nfolds10_knn3/incrs_sampletr-results.pkl', 'rb') as fileid:
    scores10_5G = pickle.load(fileid)

plt.figure()
iter_range = np.arange(0, scores10_5G["mean_scores"][:, 0].shape[0]*3, 3)
plt.plot(get_mv_avg(scores10_5G["mean_scores"][:, 0]), label='training')
plt.plot(get_mv_avg(scores10_5G["mean_scores"][:, 1]), label='test')
labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
single_components=[]
for em_id in range(6):
    tmp_sc = []
    for matrix in scores10_5G["mean_cm"]:
        tmp_sc.append(matrix[em_id, em_id] / np.sum(matrix[em_id, :]))
    single_components.append(tmp_sc)
single_components = np.array(single_components)
for j in range(single_components.shape[0]):
    plt.plot(get_mv_avg(single_components[j,:]), '--', label=f"{labels[j]}")

plt.legend()
plt.show()
