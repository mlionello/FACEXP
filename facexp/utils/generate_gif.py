import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import h5py
from pathlib import Path


def create_gif(file_in, out_folder):
    with h5py.File(file_in) as data_in:
        vertices = np.array(data_in["v"])
    fig, ax = plt.subplots(figsize=(5, 5))
    miny = np.min(vertices[:,:,1])
    maxy = np.max(vertices[:,:,1])
    minx = np.min(vertices[:,:,0])
    maxx = np.max(vertices[:,:,0])


    def update(i):
        ax.clear()
        ax.scatter(vertices[i, :, 0], vertices[i, :, 1])
        plt.xlim(minx-5, maxx+5)
        plt.ylim(miny-5, maxy+5)

    ani = animation.FuncAnimation(fig, update, frames=vertices.shape[0], interval=1)
    ani.save(out_folder / (file_in.stem + ".gif"), writer="pillow")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create vertices animation from given h5 file"
    )
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--out", type=str, default="./", help="output folder")
    args = parser.parse_args()
    create_gif(Path(args.input), Path(args.out))
