import tixelbox as tb
import tixelbox.pose_detection as posedet

with tb.sample_video() as video:
    frames = [0, 100, 200]
    poses = posedet.detect_poses(video, frames)
    posedet.draw_poses(video, poses, frames, path='sample_poses.mp4')
    print('Wrote video with poses drawn to {}'.format(os.path.abspath('sample_poses.mp4')))
