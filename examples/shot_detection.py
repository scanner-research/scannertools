import tixelbox as tb
import tixelbox.shot_detection as shotdet
import os

with tb.sample_video(delete=False) as video:
    shots = shotdet.detect_shots(video)
    montage_img = video.montage(shots)
    tb.imwrite('sample_shots.jpg', montage_img)
    print('Wrote shot montage to {}'.format(os.path.abspath('sample_shots.jpg')))
