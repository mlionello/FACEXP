#!/usr/bin/bash

mainpath=/data1/EMOVIE_sampaolo/FACE/JoJoRabbit/samplevideo/ADFES-original-stimuli/
infold=(Mediterranean_mpeg/ NorthEuropean_mpeg/)
outpath=/data1/EMOVIE_sampaolo/FACE/JoJoRabbit/medusa/mediapipe/
outfold=(ADFES_med/ ADFES_north/)

for i in 0 1; do

	outputfolder=$outpath${outfold[$i]}
	inputfolder=$mainpath${infold[$i]}

	if [ ! -d $outputfolder ]; then
		mkdir $outputfolder
	fi

	filelist=($(ls "$inputfolder"/*.mpeg))

	for infile in "${filelist[@]}"; do

		filename=$(basename "$infile")
		outfile=$outputfolder$filename
		outfile="${outfile%.*}"
		medusa_videorecon "$infile" -r mediapipe -o "$outfile".h5

	done
done
