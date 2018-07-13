import scannertools as st
import scannertools.face_detection as face_detection
import scannertools.face_embedding as face_embedding
import scannerpy
import os

with st.sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(10))
    bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames], no_execute=True)
    embeddings = face_embedding.embed_faces(db, videos=[video], frames=[frames], bboxes=bboxes, no_execute=True)
    print('First embedding: {}'.format(next(embeddings[0].load())))
    print('Finished computing embeddings')
