import scannertools as st
import scannertools.face_detection as face_detection
import scannertools.gender_detection as gender_detection
import scannerpy
import os

with st.sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(50))
    bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames])
    genders = gender_detection.detect_genders(db, videos=[video], frames=[frames], bboxes=bboxes)
    # TODO: draw genders
    print('Finished computing genders')
