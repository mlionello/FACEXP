#!/usr/bin/bash

homepath="/home/matteo.lionello/Pictures/"

for filename in "$homepath"/*; do
    if [ -d "$filename" ] || [[ "$filename" != *.jpg ]] && [[ "$filename" != *.png ]]; then
        continue
    fi
    file_stem=$(echo "$filename" | sed -r "s/.+\/(.+)\..+/\1/")
    echo "$filename"
    python ../vid2h5/get_h5_from_folder.py --input "$filename" --out ./"$file_stem".jpg --model img2vid
    ffmpeg -loop 1 -i ./"$file_stem".jpg -t 1 "$file_stem".mp4
    python ../vid2h5/get_h5_from_folder.py --input "$file_stem".mp4 --out ./tmp
done
