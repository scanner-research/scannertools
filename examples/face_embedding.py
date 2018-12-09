from scannertools import face_detection, face_embedding, sample_video
import scannerpy
import os

def main():
    with sample_video() as video:
        db = scannerpy.Database()
        frames = list(range(10))
        bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames])
        embeddings = face_embedding.embed_faces(db, videos=[video], frames=[frames], bboxes=bboxes)
        print('First embedding: {}'.format(next(embeddings[0].load())))
        print('Finished computing embeddings')

if __name__ == "__main__":
    main()
