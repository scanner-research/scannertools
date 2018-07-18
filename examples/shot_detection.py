from scannertools import shot_detection, sample_video, imwrite, Video
import scannerpy
import os

video = Video('die_hard_1988.mp4')
#with sample_video() as video:
db = scannerpy.Database()
[shots] = shot_detection.detect_shots(db, videos=[video])
montage_img = video.montage(shots)
imwrite('sample_shots.jpg', montage_img)
print('Wrote shot montage to {}'.format(os.path.abspath('sample_shots.jpg')))
