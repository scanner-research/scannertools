from .prelude import *
import numpy as np
from scannerpy import FrameType
import pickle


@scannerpy.register_python_op(name='AverageVolume')
def average_volume(config, audio: FrameType) -> bytes:
    return pickle.dumps(np.average(audio))


class AverageVolumePipeline(Pipeline):
    job_suffix = 'avgvolume'
    base_sources = ['audio']
    run_opts = {'pipeline_instances_per_node': 1}
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)

    def build_pipeline(self):
        return {'avgvolume': self._db.ops.AverageVolume(audio=self._sources['audio'].op)}

    def build_sink(self):
        return BoundOp(
            op=self._db.sinks.Column(columns=self._output_ops),
            args=[
                '{}_{}'.format(arg['path'], self.job_suffix) for arg in self._sources['audio'].args
            ])


compute_average_volume = AverageVolumePipeline.make_runner()
