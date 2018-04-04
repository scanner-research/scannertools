import tixelbox as tb
import tixelbox.object_detection as objdet

with tb.sample_video() as video:
    frames = list(range(0, video.num_frames(), 3))
    bboxes = objdet.detect_objects(video, frames)
    objdet.draw_bboxes(video, bboxes, frames, path='sample_bboxes.mp4')
