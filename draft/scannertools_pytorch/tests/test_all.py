from scannerpy import NamedVideoStream, CacheMode, NamedStream, PerfParams, DeviceType
from scannertools_infra.tests import sc
import scannertools_pytorch
import scannertools.imgproc
import scannertools_pytorch.resnet
from timeit import default_timer as now

def run(sc, op, name, device):
    vid = NamedVideoStream(sc, 'test3')
    inp = sc.io.Input([vid])
    f = sc.streams.Gather(inp, [list(range(10000))])
    res = sc.ops.Resize(frame=inp, width=[224], height=[224], device=device, batch=100)
    tf = op(frame=res, batch=100, device=device)
    out = NamedStream(sc, 'qq')
    outp = sc.io.Output(tf, [out])

    s = now()
    sc.run(
        outp,
        PerfParams.manual(500, 10000, pipeline_instances_per_node=1),
        cache_mode=CacheMode.Overwrite,
        gpu_pool='8G')
    sc.table('qq').profiler().write_trace('{}.tar.gz'.format(name))
    print('{:.1f}s'.format(now() - s))

def test_tf(sc):
    sc.ingest_videos([('test3', 'test_clip.mp4')])
    device = DeviceType.GPU
    #run(sc, sc.ops.Resnet, 'torch_python', device)
    run(sc, sc.ops.PyTorch, 'torch_cpp', device)
