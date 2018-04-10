import tixelbox as tb
import os

with tb.sample_video() as video:
    frame_nums = list(range(0, video.num_frames(), 100))
    montage_img = video.montage(frame_nums, cols=5)
    tb.imwrite('sample_montage.jpg', montage_img)
    print('Wrote frame montage to {}'.format(os.path.abspath('sample_montage.jpg')))
