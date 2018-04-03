from tixelbox import object_detection as objdet
import tixelbox as tb

with tb.sample_video(delete=False) as video:
    bboxes = objdet.detect_objects(video)
    objdet.draw_bboxes(video, bboxes, path='/tmp/test.mp4')
