import tixelbox as tb
import tixelbox.face_detection as facedet
import tixelbox.vis as vis
import scannerpy
import os

with tb.sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(50))
    bboxes = facedet.detect_faces(db, video, frames=frames)
    vis.draw_bboxes(db, video, bboxes, frames=frames, path='sample_faces.mp4')
    print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_faces.mp4')))
