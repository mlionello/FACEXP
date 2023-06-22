import python-ffmpeg
import numpy as np
import medusa.recon
import cv2


if __name__=='__main__':
    in_file = '~/Downloads/archive/train/fear/Training_135069.jpg'

    img_init = cv2.imread(in_file)
    img = cv2.resize(img_init, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)

    inimage = ffmpeg.input(img)
    out = ffmpeg.output(inimage, 'out.mp4')

