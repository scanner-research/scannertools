import scannertools as st
import scannertools.object_detection as objdet
import scannertools.vis as vis
import scannerpy
import os

with st.sample_video(delete=False) as video:
    db = scannerpy.Database()
    frames = None  #list(range(0, video.num_frames(), 3))
    bboxes = objdet.detect_objects(db, video, frames)
    vis.draw_bboxes(db, video, bboxes, frames, path='sample_objects.mp4')
    print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_objects.mp4')))
