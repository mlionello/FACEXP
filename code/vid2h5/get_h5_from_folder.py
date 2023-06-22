import os
import argparse
from medusa.recon import videorecon
from pathlib import Path
import traceback
import cv2
import sys

def get_h5(input_path, output_path, model):
	input_path = Path(input_path)
	output_path = Path(output_path)
	log_file = output_path / "error_log.txt"
	if not output_path.exists():
		output_path.mkdir(parents=True)

	for file_name in input_path.glob('**/*.mp4'):
		output_file = output_path / (file_name.stem + '.h5')
		if os.path.isfile(output_file):
			continue
		print("encoding video from: " + str(file_name) + " to: " + str(output_file))

		try:
			data = videorecon(file_name, recon_model=model, loglevel='WARNING')
			data.to_local()
			data.save(output_file)
		except Exception as e:
			print( "ERROR IN PROCESSING " + str(file_name) + "\n")
			error_message = f"Error: {e}"
			traceback_info = traceback.format_exc()
			with open(log_file, "a") as file:
				file.write("ERROR IN PROCESSING " + str(file_name) + "\n")
				file.write(error_message + "\n")
				file.write(traceback_info + "\n\n")
	return

def resampleimage(input_path, output_path):
	img_init = cv2.imread(input_path)
	img = cv2.resize(img_init, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
	cv2.imwrite(output_path, img)
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process vid files to .h5 (local)')
	parser.add_argument('--input', type=str, help='input_folder')
	parser.add_argument('--out', type=str, default='./', help='output folder')
	parser.add_argument('--model', type=str, default='mediapipe', help='videorecon model')
	args = parser.parse_args()
	get_h5(args.input, args.out, args.model)