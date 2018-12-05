from .prelude import *
import numpy as np
from scannerpy import FrameType
import pickle

import json
@scannerpy.register_python_op('PrintCap')
def print_cap(config, captions: bytes) -> bytes:
    print(json.loads(captions.decode('utf-8')))
    return b' '

class PrintCaptionsPipeline(Pipeline):
    job_suffix = 'printcap'
    base_sources = ['captions']
    run_opts = {'pipeline_instances_per_node': 1}
    parser_fn = lambda _: lambda buf, _: ()

    def build_pipeline(self):
        return {
            'cap': self._db.ops.PrintCap(captions=self._sources['captions'].op)
        }

    def build_sink(self):
        return BoundOp(
            op=self._db.sinks.Column(columns=self._output_ops),
            args=[
                '{}_{}'.format(arg['path'], self.job_suffix)
                for arg in self._sources['captions'].args
            ])

print_captions = PrintCaptionsPipeline.make_runner()
