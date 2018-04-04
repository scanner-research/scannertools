import pytest
import tixelbox as tb
import os


@pytest.fixture(scope='module')
def video():
    with tb.sample_video() as video:
        yield video


@pytest.fixture(scope='module')
def audio(video):
    audio = video.audio()
    yield audio
    os.remove(audio.path())


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
