from scannertools import face_detection, vis, sample_video
import scannerpy
import os

with sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(50))
    bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames])
    vis.draw_bboxes(db, videos=[video], frames=[frames], bboxes=bboxes, paths=['sample_faces.mp4'])
    print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_faces.mp4')))
