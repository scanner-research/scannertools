import tixelbox as tb
import numpy as np
import cv2

with tb.sample_video() as video:
    frame_nums = list(range(0, video.num_frames(), 100))
    montage_img = video.montage(frame_nums, cols=5)
    tb.imwrite('montage.jpg', montage_img)
