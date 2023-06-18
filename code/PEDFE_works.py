
import pandas as pd
from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import numpy as np
import sys

df = pd.read_excel(r'../../samplevideo/PEDFE/Supplemental_Material_T2.xlsx');
first_row_index = df.first_valid_index()
column_names = df.iloc[first_row_index].tolist()
data_rows = df.iloc[first_row_index + 1:]

in_folder = Path("../mediapipe/PEDFE/")
# Get the mean expression across the dataset across equal distributed expressions
ref_mean = None
vertices = []
print(in_folder)
for j, fileh5 in enumerate(in_folder.glob("*.h5")):
	print(str(j), end = '\r')
	with h5py.File(fileh5) as data_in:
		vertices = [vertices, data_in["v"]] # (t_points, 468, 3),
		if ref_mean is None:
			ref_mean = np.mean(vertices[-1],0)[:,:,None]
		else:
			ref_mean = np.concatenate((ref_mean, np.mean(data_in["v"],0)[:,:,None]), axis = 2)

	if j ==10:
		break

ref_mean = np.mean(ref_mean,2)
pdist_ref_mean = pdist(ref_mean)
features = None
for k, vertix in enumerate(vertices):
	print(vertices[k])
	print(vertices[k].shape)
	if features is None:
		features = np.array(pdist(np.mean(vertices[k], 0))-pdist_ref_mean)[:,None]
	else:
		features = np.concatenate((features, np.array(pdist(np.mean(mean_ref[k],0))-pdist_ref_mean)[:,None]), axis=1)
print(features.shape)

