from scannertools import pose_detection, vis, sample_video
import scannerpy
import os

with sample_video() as video:
    db = scannerpy.Database()
    frames = [0, 100, 200]
    poses = pose_detection.detect_poses(db, video, frames)
    vis.draw_poses(db, video, poses, frames, path='sample_poses.mp4')
    print('Wrote video with poses drawn to {}'.format(os.path.abspath('sample_poses.mp4')))
