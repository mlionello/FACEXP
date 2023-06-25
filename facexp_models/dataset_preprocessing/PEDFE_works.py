import pandas as pd
from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import numpy as np

df = pd.read_excel(r"../../Supplemental_Material_T2.xlsx")
first_row_index = df.first_valid_index()
column_names = df.iloc[first_row_index].tolist()
print("available entries: ", column_names)
data = df.iloc[first_row_index + 1 :].reset_index(drop=True)
data.columns = column_names

in_folder = Path("../../mediapipe/PEDFE/")
# Get the mean expression across the dataset across equal distributed expressions
mesh_init_mean = []
mesh_mean = []
file_ids = []
for j, fileh5 in enumerate(in_folder.glob("*.h5")):
    print(str(j), end="\r")
    with h5py.File(fileh5) as data_in:
        file_ids.append(fileh5)
        # Concatenating the temporal mean from each 3D-mesh
        mesh_init_mean.append(np.mean(data_in["v"], 0))
        mesh_mean.append(np.mean(data_in["v"], 0))

ref_mean = np.mean(mesh_init_mean, 0)
pdist_ref_mean = pdist(ref_mean)

features = []
for j, mesh in enumerate(mesh_mean):
    print(str(j), end="\r")
    features.append(np.array(pdist(mesh) - pdist_ref_mean))

features = np.array(features)

# Aligning info entries with filename of the vertices
valid_data_indices = []
for x in file_ids:
    tmp = np.where(x.stem == data.loc[:, "PEDFE_code"])[0]
    if len(tmp) != 0:
        valid_data_indices.append(tmp)

valid_data = data.iloc[np.reshape(valid_data_indices, -1)].reset_index(drop=True)

np.save("features", features)
np.save("labels", valid_data)
