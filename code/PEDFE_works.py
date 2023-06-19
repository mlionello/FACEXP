import pandas as pd
from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import numpy as np

df = pd.read_excel(r'../samplevideo/PEDFE/Supplemental_Material_T2.xlsx')
first_row_index = df.first_valid_index()
column_names = df.iloc[first_row_index].tolist()

print("available entries: ", column_names)
data = df.iloc[first_row_index + 1:].reset_index(drop=True)
data.columns = column_names

in_folder = Path("../mediapipe/PEDFE/")
# Get the mean expression across the dataset across equal distributed expressions
ref_mean_subj = None
vertices = []
file_ids = []
print(in_folder)
for j, fileh5 in enumerate(in_folder.glob("*.h5")):
    print(str(j), end='\r')
    with h5py.File(fileh5) as data_in:
        file_ids.append(fileh5)
        vertices.append(np.array(data_in["v"]))  # (t_points, 468, 3),
        if ref_mean_subj is None:
            ref_mean_subj = np.mean(vertices[-1], 0)[:, :, None]
        else:
            ref_mean_subj = np.concatenate((ref_mean_subj, np.mean(data_in["v"], 0)[:, :, None]), axis=2)

# align info entries with vertices
valid_data_indices = []
for x in file_ids:
    tmp = np.where(x.stem == data.loc[:, "PEDFE_code"])[0][0]
    valid_data_indices.append(tmp)
valid_data = data.loc[valid_data_indices].reset_index(drop=True)

ref_mean = np.mean(ref_mean_subj, 2)

pdist_ref_mean = pdist(ref_mean)
features = None
for k, vertix in enumerate(vertices):
    if features is None:
        features = np.array(pdist(np.mean(vertix, 0)) - pdist_ref_mean)[:, None]
    else:
        features = np.concatenate((features, np.array(pdist(np.mean(vertix, 0)) - pdist_ref_mean)[:, None]),
                                  axis=1)

np.save('features', features)
np.save('labels', valid_data)

