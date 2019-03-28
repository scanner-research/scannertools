from attr import attrs, attrib, evolve
from .video import Video
from collections import namedtuple, defaultdict
import scannerpy
import inspect
from scannerpy import FrameType
from typing import Generic, TypeVar, Any, List


T = TypeVar('T')

class StreamParam(Generic[T]):
    pass

class Stream(Generic[T]):
    pass

class Sink(Generic[T]):
    pass

class IO:
    pass

@attrs(frozen=True)
class Input(IO):
    type = attrib()
    default = attrib(default=None)

@attrs(frozen=True)
class Output(IO):
    type = attrib()

def has_type(ty, base):
    # TODO: what's the right way to do this?
    return base in ty.__mro__

@attrs(frozen=True)
class BoundVar:
    cls = attrib()
    var = attrib()
    key = attrib()

    def resolve(self):
        if isinstance(self.var, Input):
            ty = self.var.type
            cls_args = self.cls.args()
            if has_type(ty, Stream):
                return cls_args[self.key].resolve()
            elif has_type(ty, StreamParam):
                if self.key in cls_args:
                    return cls_args[self.key].resolve()
                else:
                    return self.cls._stream_params[self.key]
            else:
                raise Exception(ty)
        elif isinstance(self.var, Output):
            return self.cls._outputs[self].resolve()
        else:
            raise Exception("Unreachable")

class OpGraph():
    def __init__(self, **kwargs):
        for k, v in vars(type(self)).items():
            if isinstance(v, IO):
                setattr(self, k, BoundVar(cls=self, key=k, var=v))

        self._args = kwargs
        self._expanded = False

    def args(self):
        return self._args

    def initialize(self, db):
        self.db = db

    def op_param_to_op(self, op):
        if isinstance(op, OpLeaf):
            return op
        else:
            return op.cls

    def expand_graph(self):
        if self._expanded:
            return
        else:
            self._expanded = True

        outputs = self.build()
        self._outputs = outputs

        nodes = self.graph_search(outputs.values())
        edges = set()
        for node in nodes:
            for other_node in node.args().values():
                if isinstance(other_node, BoundVar) and isinstance(other_node.var, Output):
                    edges.add((self.op_param_to_op(other_node), node))

        self._sorted_nodes = self.toposort(nodes, edges)

        for node in self._sorted_nodes:
            node.initialize(self.db)
            node.expand_graph()

        return outputs

    def graph_search(self, ops):
        # Breadth-first search
        frontier = set(ops)
        visited = set()
        while len(frontier) > 0:
            visited |= frontier
            new_frontier = set()
            for node in frontier:
                for possible_node in self.op_param_to_op(node).args().values():
                    if isinstance(possible_node, BoundVar) and isinstance(possible_node.var, Output):
                        new_frontier.add(possible_node)
            frontier = new_frontier

        return {self.op_param_to_op(bound_output) for bound_output in visited}

    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    def toposort(self, nodes, edges):
        sorted_list = []

        # Make a copy of edges so we can delete from it
        edges = list(edges)

        # Initialize with nodes that have no incoming edges
        to_visit = [node for node in nodes
                    if all([edge[1] != node for edge in edges])]

        while len(to_visit) > 0:
            parent = to_visit.pop()
            sorted_list.append(parent)

            for child in nodes:
                to_child_edges = [i for i, edge in enumerate(edges)
                                  if edge[0] == parent and edge[1] == child]

                if len(to_child_edges) == 0:
                    continue

                # Remove edges in reverse order to preserve correct indices
                for i in to_child_edges[::-1]:
                    del edges[i]

                if all([edge[1] != child for edge in edges]):
                    to_visit.append(child)

        if len(edges) > 0:
            raise Exception("Block graph contains a cycle")

        return sorted_list

    def job_args(self):
        pass

    def build_jobs(self, stream_params_args, job_args, N):
        for node in self._sorted_nodes:
            node_stream_params_args = {
                getattr(node, k): (stream_params_args[arg])
                for k, arg in node.args().items() if has_type(arg.var.type, StreamParam)
            }
            if len(node_stream_params_args) > 0:
                node.build_jobs(node_stream_params_args, job_args, N)


        job_args_builder_params = list(inspect.signature(self.job_args).parameters.keys())
        if len(job_args_builder_params) > 0:
            stream_params_arg_names = \
                [k for k, v in vars(type(self)).items()
                 if isinstance(v, Input) and has_type(v.type, StreamParam)]

            var_to_str_map = {v.key: v for v in stream_params_args.keys()}

            for i in range(N):
                named_args = {name: stream_params_args[var_to_str_map[name]][i] for name in job_args_builder_params}
                job_args[i].update({k.resolve(): v for k, v in self.job_args(**named_args).items()})

    def run(self, db, stream_params, globals={}):
        self._stream_params = stream_params
        self.initialize(db)
        self.expand_graph()

        output_ops = [getattr(self, k).resolve() for k, var in vars(type(self)).items() if isinstance(var, Output)]
        assert(len(output_ops) == 1)

        assert(len(stream_params) >= 1)
        N = len(stream_params)
        job_args = [{} for _ in range(len(stream_params))]
        stream_params_soa = {k: [s[k] for s in stream_params] for k in stream_params[0].keys()}
        self.build_jobs(stream_params_soa, job_args, N)
        jobs = [scannerpy.Job(op_args=op_args) for op_args in job_args]

        db.run(output_ops[0], jobs, force=True)

class OpLeaf(OpGraph):
    def __init__(self, f, args={}):
        self._f = f
        self._args = args

    def args(self):
        return self._args

    def expand_graph(self):
        for key, node in self._args.items():
            self.op_param_to_op(node).expand_graph()

        self.val = self._f(**{key: node.resolve() for key, node in self._args.items()})

    def resolve(self):
        return self.val


class ScannerFrameSource(OpGraph):
    video = Input(StreamParam[Video])
    frame = Output(Stream[FrameType])

    def build(self):
        return {
            self.frame: OpLeaf(lambda: self.db.sources.FrameColumn())
        }

    def job_args(self, video):
        return {self.frame: self.db.table(video.scanner_name()).column('frame')}


class Gather(OpGraph):
    input_stream = Input(Stream[Any])
    indices = Input(StreamParam[List[int]])
    output_stream = Output(Stream[Any])

    def build(self):
        return {
            self.output_stream: OpLeaf(lambda stream: self.db.streams.Gather(stream),
                                       {'stream': self.input_stream})
        }

    def job_args(self, indices):
        return {self.output_stream: indices}


class GatheredFrame(OpGraph):
    video = Input(StreamParam[Video])
    indices = Input(StreamParam[List[int]])
    frame = Output(Stream[FrameType])
    stream = Output(Stream[FrameType])

    def build(self):
        frame_source = ScannerFrameSource(video=self.video)
        gather = Gather(input_stream=frame_source.frame, indices=self.indices)
        return {
            self.frame: frame_source.frame,
            self.stream: gather.output_stream
        }


class Histogram(OpGraph):
    frame = Input(Stream[FrameType])
    bins = Input(int, default=16)
    histogram = Output(Stream[bytes])

    def build(self):
        print(self.bins)
        return {
            self.histogram: OpLeaf(lambda frame: self.db.ops.Histogram(frame=frame),
                                   {'frame': self.frame})
        }


class ScannerColumnSink(OpGraph):
    column = Input(Stream[Any])
    table_name = Input(StreamParam[str])
    sink = Output(Sink[Any])

    def build(self):
        return {
            self.sink: OpLeaf(lambda column: self.db.sinks.Column(columns={'column': column}),
                              {'column': self.column})
        }

    def job_args(self, table_name):
        return {self.sink: table_name}


class HistogramPipeline(OpGraph):
    video = Input(StreamParam[Video])
    table_name = Input(StreamParam[str])
    indices = Input(StreamParam[List[int]])
    histograms = Output(Sink[bytes])

    def build(self):
        frame = GatheredFrame(video=self.video, indices=self.indices)
        histogram = Histogram(frame=frame.stream)
        sink = ScannerColumnSink(column=histogram.histogram, table_name=self.table_name)
        return {self.histograms: sink.sink}


from .prelude import sample_video
with sample_video(delete=False) as video:
    db = scannerpy.Database()

    p = HistogramPipeline()
    result = p.run(
        db,
        stream_params=[{p.video: video, p.table_name: "test", p.indices: [0]}])

    print(result[0].histograms.load())
