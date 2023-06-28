from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import numpy as np
import re

in_folder = Path("../../mediapipe/pilot/")
# Get the mean expression across the dataset across equal distributed expressions
mesh_init_mean = []
mesh_mean = []
file_ids = []
labels = []
digit_pattern = r'\d{2}'
for j, fileh5 in enumerate(in_folder.glob("*.h5")):
    print(str(j), end="\r")
    with h5py.File(fileh5) as data_in:
        file_ids.append(fileh5)
        digits = re.findall(digit_pattern, fileh5.stem)
        labels.append([int(digit) for digit in digits])

        # Concatenating the temporal mean from each 3D-mesh
        mesh_init_mean.append(np.mean(data_in["v"], 0))
        mesh_mean.append(np.mean(data_in["v"], 0))

labels = np.array(labels)
ref_mean = np.mean(mesh_init_mean, 0)
pdist_ref_mean = pdist(ref_mean)

features = []
for j, mesh in enumerate(mesh_mean):
    print(str(j), end="\r")
    features.append(np.array(pdist(mesh) - pdist_ref_mean))

features = np.array(features)

np.save("features_pilot", features)
np.save("labels_pilot", labels)
