import scannertools as st
import scannertools.pose_detection as posedet
import scannertools.vis as vis
import scannerpy
import os

with st.sample_video() as video:
    db = scannerpy.Database()
    frames = [0, 100, 200]
    poses = posedet.detect_poses(db, video, frames)
    vis.draw_poses(db, video, poses, frames, path='sample_poses.mp4')
    print('Wrote video with poses drawn to {}'.format(os.path.abspath('sample_poses.mp4')))
