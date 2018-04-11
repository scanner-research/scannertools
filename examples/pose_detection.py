import tixelbox as tb
import tixelbox.pose_detection as posedet
import scannerpy
import os

with tb.WithMany(scannerpy.Database(), tb.sample_video()) as (db, video):
    frames = [0, 100, 200]
    poses = posedet.detect_poses(db, video, frames)
    posedet.draw_poses(db, video, poses, frames, path='sample_poses.mp4')
    print('Wrote video with poses drawn to {}'.format(os.path.abspath('sample_poses.mp4')))
