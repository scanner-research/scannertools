from scannertools import face_detection, clothing_detection, sample_video
import scannerpy
import os

with sample_video(delete=False) as video:
    db = scannerpy.Database()
    frames = [0, 100, 200]
    [bboxes] = face_detection.detect_faces(db, videos=[video], frames=[frames], no_execute=True)
    [clothing] = clothing_detection.detect_clothing(db, videos=[video], frames=[frames], bboxes=[bboxes])
    print(next(clothing.load())[0])
