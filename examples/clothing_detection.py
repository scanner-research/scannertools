from scannertools import face_detection, clothing_detection, sample_video
import scannerpy
import os

def main():
    with sample_video() as video:
        db = scannerpy.Database()
        frames = [0, 100, 200]
        [bboxes] = face_detection.detect_faces(db, videos=[video], frames=[frames])
        [clothing] = clothing_detection.detect_clothing(db, videos=[video], frames=[frames], bboxes=[bboxes])
        print(next(clothing.load())[0])

if __name__ == "__main__":
    main()
