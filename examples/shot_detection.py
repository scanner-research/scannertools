from scannertools import shot_detection, sample_video, imwrite
import scannerpy
import os

with sample_video() as video:
    db = scannerpy.Database()
    shots = shot_detetection.detect_shots(db, video)
    montage_img = video.montage(shots)
    imwrite('sample_shots.jpg', montage_img)
    print('Wrote shot montage to {}'.format(os.path.abspath('sample_shots.jpg')))
