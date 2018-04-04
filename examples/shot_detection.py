import tixelbox as tb
import tixelbox.shot_detection as shotdet

with tb.sample_video(delete=False) as video:
    shots = shotdet.detect_shots(video)
    montage_img = video.montage(shots)
    tb.imwrite('shots.jpg', montage_img)
