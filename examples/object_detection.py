from scannertools import sample_video, object_detection, vis
import scannerpy
import os

def main():
    with sample_video() as video:
        db = scannerpy.Database()
        frames = list(range(0, 20, 3))

        print('Running object detector')
        [bboxes] = object_detection.detect_objects(db, videos=[video], frames=[frames])

        print('Running bbox visualizer')
        vis.draw_bboxes(db, videos=[video], frames=[frames], bboxes=[bboxes], paths=['sample_objects.mp4'])

        print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_objects.mp4')))

if __name__ == "__main__":
    main()
