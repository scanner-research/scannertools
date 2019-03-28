from scannerpy import NamedVideoStream, CacheMode, NamedStream, PerfParams, DeviceType
from scannertools_infra.tests import sc
import scannertools_tfcpp
import scannertools.object_detection
from timeit import default_timer as now

def run(sc, op, name):
    vid = NamedVideoStream(sc, 'test1')
    inp = sc.io.Input([vid])
    #f = sc.streams.Gather(inp, [list(range(1000))])
    tf = op(frame=inp, batch=100, device=DeviceType.CPU)
    out = NamedStream(sc, 'qq')
    outp = sc.io.Output(tf, [out])

    s = now()
    sc.run(outp, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, pipeline_instances_per_node=1)
    sc.table('qq').profiler().write_trace('{}.trace'.format(name))
    print('{:.1f}s'.format(now() - s))

def test_tf(sc):
    run(sc, sc.ops.DetectObjects, 'python')
    run(sc, sc.ops.TFC, 'cpp')
