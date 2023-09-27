from medusa.render import VideoRenderer
from pathlib import Path
import argparse
import os
import h5py


def main(inputfile, outfolder, invid, shading):
    inputfile = Path(inputfile)
    outfolder = Path(outfolder)

    if not inputfile.exists():
        inputfile.mkdir(parents=True)
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    on_bckg = ""
    if vid is not None:
        on_bckg="_ol"
    f_out = outfolder / 'render' / f"{inputfile.stem}_{shading}{on_bckg}.mp4"
    if fout.exists():
        print(f"output file already exists: {fout}")
        return

    # data_4d.apply_vertex_mask('face')
    with h5py.File(inputfile) as data_in:
        data_4d = data_in("v")

    if vid is not None:
        renderer = VideoRenderer(shading=shading, loglevel='INFO', background=invid)
    else:
        renderer = VideoRenderer(shading=shading, loglevel='INFO')
    renderer.render(f_out, data_4d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="render h5 mask on original video")
    parser.add_argument("--input", type=str, help="input_file")
    parser.add_argument("--invid", default=None, type=str, help="input_vid")
    parser.add_argument("--out", type=str, default="./", help="output folder")
    parser.add_argument(
        "--shading", type=str, default="flat", help="Type of shading ('flat', 'smooth', or 'wireframe'; latter only "
                                                    "when using 'pyrender')"
    )
    args = parser.parse_args()
    main(args.input, args.out, args.invid, args.shading)
