import argparse
import os

from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import numpy as np

def get_dataset(inpath, outfolder, refh5):
    if not outfolder.is_dir():
        os.mkdir(outfolder)
    column_names = ["Modality", "Channel", "Emotion", "Intensity",
                    "Statement", "Repetition", "Actor"]
    print("available entries: ", column_names)

    meshatrest = np.load(refh5)
    pdist_rest = pdist(meshatrest)

    # Get the mean expression across the dataset across equal distributed expressions
    file_ids = []
    features = []
    labels = []
    for j, fileh5 in enumerate(inpath.glob("*.h5")):
        print(str(j), end='\r')
        with h5py.File(fileh5) as data_in:
            data_row = fileh5.stem.split('-')
            labels.append([int(val) for val in data_row])
            file_ids.append('/'.join(fileh5.parts[-2:]))
            features.append(pdist(np.mean(data_in["v"], 0)) - pdist_rest)

    labels = {"file_ids": file_ids, "labels": labels, "column_names": column_names}
    np.save( outfolder / 'features', features)
    np.save( outfolder / 'labels', labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='h5 to dataframes and npy for features and labels')
    parser.add_argument('--input', type=str, help='input_folder')
    parser.add_argument('--ref', type=str, default='./refAtRest.npy', help='ref h5 template at rest')
    parser.add_argument('--out', type=str, default='./', help='output folder')
    args = parser.parse_args()
    get_dataset(Path(args.input), Path(args.out), Path(args.ref))
