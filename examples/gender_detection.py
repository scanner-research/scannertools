from scannertools import face_detection, gender_detection, sample_video
import scannerpy
import os

def main():
    with sample_video() as video:
        db = scannerpy.Database()
        frames = list(range(50))
        bboxes = face_detection.detect_faces(db, videos=[video], frames=[frames])
        genders = gender_detection.detect_genders(db, videos=[video], frames=[frames], bboxes=bboxes)
        print('First gender: {}'.format(next(genders[0].load())))
        print('Finished computing genders')

if __name__ == "__main__":
    main()
