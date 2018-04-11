import tixelbox as tb
import tixelbox.shot_detection as shotdet
import scannerpy
import os

with tb.WithMany(scannerpy.Database(), tb.sample_video()) as (db, video):
    shots = shotdet.detect_shots(db, video)
    montage_img = video.montage(shots)
    tb.imwrite('sample_shots.jpg', montage_img)
    print('Wrote shot montage to {}'.format(os.path.abspath('sample_shots.jpg')))
