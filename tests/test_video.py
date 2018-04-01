import pytest
import tixelbox as tb
import tempfile
import requests
import os


@pytest.fixture(scope='module')
def video():
    url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        f.flush()
        yield tb.Video(f.name)


@pytest.fixture(scope='module')
def audio(video):
    audio = video.audio()
    yield audio
    os.remove(audio.path())


def test_frame(video):
    frame = video.frame(0)
    assert frame.shape == (480, 640, 3)


def test_frame_time(video):
    video.frame(time=10)


def test_frames(video):
    frames = video.frames([0, 1])
    assert len(frames) == 2


def test_audio(video, audio):
    path = audio.extract()
    assert os.path.isfile(path)
    os.remove(path)
