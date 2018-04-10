import tixelbox as tb
import tixelbox.object_detection as objdet
import tixelbox.vis as vis
import scannerpy
import os

with tb.WithMany(scannerpy.Database(), tb.sample_video()) as (db, video):
    frames = list(range(0, video.num_frames(), 3))
    bboxes = objdet.detect_objects(db, video, frames)
    vis.draw_bboxes(db, video, bboxes, frames, path='sample_objects.mp4')
    print('Wrote video with objects drawn to {}'.format(os.path.abspath('sample_objects.mp4')))
