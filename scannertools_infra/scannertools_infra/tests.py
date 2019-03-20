from scannerpy import Config, Client
import pytest
import tempfile
import socket
import requests
import toml
from subprocess import check_call as run
import subprocess
import GPUtil


needs_gpu = pytest.mark.skipif(len(GPUtil.getGPUs()) == 0, reason='need GPU to run')


def make_config(master_port=None, worker_port=None, path=None):
    cfg = Config.default_config()
    cfg['network']['master'] = 'localhost'
    cfg['storage']['db_path'] = tempfile.mkdtemp()
    if master_port is not None:
        cfg['network']['master_port'] = master_port
    if worker_port is not None:
        cfg['network']['worker_port'] = worker_port

    if path is not None:
        with open(path, 'w') as f:
            cfg_path = path
            f.write(toml.dumps(cfg))
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(bytes(toml.dumps(cfg), 'utf-8'))
    return (cfg_path, cfg)


def download_videos():
    # Download video from GCS
    url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        host = socket.gethostname()
        # HACK: special proxy case for Ocean cluster
        if host in ['ocean', 'crissy', 'pismo', 'stinson']:
            resp = requests.get(
                url,
                stream=True,
                proxies={'https': 'http://proxy.pdl.cmu.edu:3128/'})
        else:
            resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        vid1_path = f.name

    # Make a second one shorter than the first
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        vid2_path = f.name
    run([
        'ffmpeg', '-y', '-i', vid1_path, '-ss', '00:00:00', '-t', '00:00:10',
        '-c:v', 'libx264', '-strict', '-2', vid2_path
    ])

    return (vid1_path, vid2_path)

@pytest.fixture(scope="module")
def sc():
    # Create new config
    (cfg_path, cfg) = make_config()

    # Setup and ingest video
    with Client(config_path=cfg_path, debug=True) as sc:
        (vid1_path, vid2_path) = download_videos()

        sc.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        sc.ingest_videos(
            [('test1_inplace', vid1_path), ('test2_inplace', vid2_path)],
            inplace=True)

        yield sc

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])
