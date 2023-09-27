#!/usr/bin/bash

homepath="/home/matteo/Pictures/Screenshots/"

for filename in "$homepath"/*; do
    if [ -d "$filename" ] || [[ "$filename" != *.jpg ]] && [[ "$filename" != *.png ]]; then
        continue
    fi
    echo "$filename"
    python ../vid2h5/get_h5_from_folder.py --input "$filename" --out ./tmpresampled.jpg --model img2vid
    ffmpeg -loop 1 -i ./tmpresampled.jpg -t 1 tmpvidout.mp4
    python ../vid2h5/get_h5_from_folder.py --input tmpvidout.mp4 --out ./tmp
done
