from scannertools_infra.tests import sc, needs_gpu
from scannerpy.storage import NamedVideoStream, NamedStream
from scannerpy import PerfParams, DeviceType
import scannertools_caffe.pose_detection

@needs_gpu
def test_pose(sc):
    vid = [NamedVideoStream(sc, 'test1')]
    frame = sc.io.Input(vid)
    frame_sample = sc.streams.Gather(frame, [list(range(0, 1000, 100))])
    pose = sc.ops.OpenPose(
        frame=frame_sample,
        device=DeviceType.GPU,
        pose_num_scales=6,
        pose_scale_gap=0.16,
        compute_hands=True,
        hand_num_scales=6,
        hand_scale_gap=0.16,
        compute_face=True,
        batch=5
    )
    output = NamedStream(sc, 'test1-pose')
    output_op = sc.io.Output(pose, [output])

    sc.run(output_op, PerfParams.estimate())
