from scannertools import face_detection, face_embedding, sample_video
import scannerpy
import os

with sample_video() as video:
    db = scannerpy.Database()
    frames = list(range(10))
    bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames], no_execute=True)
    embeddings = face_embedding.embed_faces(db, videos=[video], frames=[frames], bboxes=bboxes, no_execute=True)
    print('First embedding: {}'.format(next(embeddings[0].load())))
    print('Finished computing embeddings')
