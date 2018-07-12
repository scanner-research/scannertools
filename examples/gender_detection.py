import scannertools as st
import scannertools.face_detection as facedet
import scannertools.gender_detection as genderdet
import scannerpy
import os

with st.sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(50))
    bboxes = facedet.detect_faces(db, videos=[video], frames=[frames])
    genders = genderdet.detect_genders(db, videos=[video], frames=[frames], bboxes=bboxes)
    # TODO: draw genders
    print('Finished computing genders')
