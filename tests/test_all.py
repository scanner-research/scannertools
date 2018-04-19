import pytest
import scannertools as st
import scannertools.object_detection as objdet
import scannertools.face_detection as facedet
import scannertools.optical_flow as optflow
import scannertools.shot_detection as shotdet
import scannertools.pose_detection as posedet
import scannerpy
import os
import subprocess as sp
import tempfile
import toml

try:
    sp.check_call(['nvidia-smi'])
    has_gpu = True
except (OSError, sp.CalledProcessError):
    has_gpu = False

needs_gpu = pytest.mark.skipif(not has_gpu, reason='need GPU to run')


@pytest.fixture(scope='module')
def video():
    with st.sample_video() as video:
        yield video


@pytest.fixture(scope='module')
def audio(video):
    audio = video.audio()
    yield audio
    os.remove(audio.path())


@pytest.fixture(scope='module')
def db():
    cfg = scannerpy.Config.default_config()
    cfg['network']['master'] = 'localhost'
    cfg['storage']['db_path'] = tempfile.mkdtemp()

    with tempfile.NamedTemporaryFile() as cfg_f:
        cfg_f.write(bytes(toml.dumps(cfg), 'utf-8'))
        cfg_f.flush()

        with scannerpy.Database(config_path=cfg_f.name) as db:
            yield db


def test_frame(video):
    frame = video.frame(0)
    assert frame.shape == (video.height(), video.width(), 3)


def test_metadata(video):
    video.fps()
    video.num_frames()
    video.duration()


def test_frame_time(video):
    video.frame(number=10)
    video.frame(time=10)


def test_frames(video):
    frames = video.frames([0, 1])
    assert len(frames) == 2


def test_audio(video, audio):
    path = audio.extract()
    assert os.path.isfile(path)
    os.remove(path)


def test_object_detection(db, video):
    bboxes = objdet.detect_objects(db, video, frames=[0], nms_threshold=0.5)
    assert len([bb for bb in bboxes[0] if bb.score > 0.5]) == 1


def test_face_detection(db, video):
    bboxes = facedet.detect_faces(db, video, frames=[0])
    assert len([bb for bb in bboxes[0] if bb.score > 0.5]) == 1


def test_montage(video):
    video.montage([0, 1], cols=2)


def test_optical_flow(db, video):
    optflow.compute_flow(db, video, frames=[1])


def test_shot_detection(db, video):
    shotdet.detect_shots(db, video)


@needs_gpu
def test_pose_detection(db, video):
    posedet.detect_poses(db, video, frames=[0])
