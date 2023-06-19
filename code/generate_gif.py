import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import h5py
from pathlib import Path


def create_gif(file_in, out_folder):
    with h5py.File(file_in) as data_in:
        vertices = np.array(data_in["v"])
    fig, ax = plt.subplots(figsize=(5,5))
    an_ind = 2
    an_data = vertices[an_ind]

    def update(i):
        ax.clear()
        ax.scatter(an_data[i,:, 0], an_data[i,:, 1])
        plt.xlim(-9, 9)
        plt.ylim(-10, 10)

    ani = animation.FuncAnimation(fig, update, frames=an_data.shape[0], interval=1)
    ani.save(out_folder / (file_in.stem + '.gif'), writer='pillow')
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create vertices animation from given h5 file')
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('--out', type=str, default='./', help='output folder')
    args = parser.parse_args()
    create_gif(Path(args.input), Path(args.out))

