import tixelbox as tb
import tixelbox.face_detection as facedet
import tixelbox.vis as vis
import os

with tb.WithMany(scannerpy.Database(), tb.sample_video()) as (db, video):
    bboxes = facedet.detect_faces(db, video)
    vis.draw_bboxes(db, video, bboxes, path='sample_faces.mp4')
    print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_faces.mp4')))
