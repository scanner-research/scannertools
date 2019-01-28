from attr import attrs, attrib, evolve
from .video import Video
from collections import namedtuple, defaultdict
import scannerpy
import inspect
from scannerpy import FrameType
from typing import Generic, TypeVar, Any


@attrs(frozen=True)
class LazyAccess:
    obj = attrib()
    key = attrib()

    def get(self):
        return getattr(self.obj, self.key)

    def resolve(self):
        next = self.get()
        if isinstance(next, LazyAccess):
            return next.resolve()
        else:
            return next

    def __getattr__(self, key):
        return LazyAccess(obj=self, key=key)


T = TypeVar('T')

class Stream(Generic[T]):
    pass

class Sink(Generic[T]):
    pass

class Input:
    def __init__(self, ty, default=None):
        self._type = ty
        self._default = default

class Output:
    def __init__(self, ty):
        self._type = ty

class OpGraphMeta(type):
    def __init__(cls, name, bases, ns):
        super(OpGraphMeta, cls).__init__(name, bases, ns)
        cls._output_keys = [k for k, v in vars(cls).items() if isinstance(v, Output)]

class OpGraph(metaclass=OpGraphMeta):
    def __init__(self, *args, **kwargs):
        self._build_args = (args, kwargs)
        self._built = False

    def __getattribute__(self, k):
        output_keys = super(OpGraph, self).__getattribute__('_output_keys')
        if k in output_keys:
            return LazyAccess(self, k)
        else:
            return super(OpGraph, self).__getattribute__(k)

    def initialize(self, db):
        self.db = db

    def _build(self, globals):
        if self._built:
            return
        else:
            self._built = True

        (args, kwargs) = self._build_args
        sig = inspect.signature(self.build)
        for param, ty in sig.parameters.items():
            ty = ty.annotation
            if Stream in ty.__mro__:
                kwargs[param] = kwargs[param].get()

        my_globals = {k.key: v for k, v in globals.items() if k.obj == self}
        bindings = self.build(*args, **kwargs, **my_globals)
        self._bindings = bindings

        #nodes = self._search_and_toposort(aliases)
        nodes = list(self._bindings.values())
        print(nodes)

        for node in nodes:
            if isinstance(node, OpLeaf):
                graph = node
            elif not isinstance(node, LazyAccess):
                raise Exception("Object {} is not a LazyAccess".format(node))
            else:
                graph = node.obj

            graph.initialize(db)
            graph._build({
                LazyAccess(k.obj.get(), k.key): v
                for k, v in globals.items() if isinstance(k.obj, LazyAccess) and k.obj.get() == node
            })

        return type(aliases)(**{k: v.get() if isinstance(v, LazyAccess) else v for k, v in aliases._asdict().items()})

    def _search_and_toposort(self, aliases):
        # TODO
        pass

    def run(self, db, outputs, per_video, globals):
        aliases = self._build(globals)
        outputs = [node.resolve() for node in outputs]
        print(outputs)

class OpLeaf:
    def __init__(self, f, args=[]):
        self._f = f
        self._args = args

    def initialize(self, db):
        self.db = db

    def _build(self, globals):
        for node in self._args:
            if isinstance(node, OpLeaf):
                graph = node
            elif not isinstance(node, LazyAccess):
                raise Exception("Object {} is not a Node".format(node))
            else:
                graph = node.obj

            graph.initialize(db)
            graph._build({})

        op = self._f(*[node.get() if isinstance(node, Node) else node._op for node in self._args])
        self._op = op
        return op


class ScannerFrameSource(OpGraph):
    frame = Output(Stream[FrameType])

    def build(self):
        return {
            self.frame: OpLeaf(lambda: self.db.sources.FrameColumn())
        }

class Gather(OpGraph):
    input_stream = Input(Stream[Any])
    output_stream = Output(Stream[Any])

    def build(self):
        return {
            self.output_stream: OpLeaf(lambda stream: self.db.streams.Gather(stream),
                                       [self.input_stream])
        }

class GatheredFrame(OpGraph):
    frame = Output(Stream[FrameType])
    stream = Output(Stream[FrameType])

    def build(self):
        frame = ScannerFrameSource()
        gather = Gather(stream=frame.frame)
        return {
            self.frame: frame.frame,
            self.stream: gather.stream
        }

class Histogram(OpGraph):
    frame = Input(Stream[FrameType])
    bins = Input(int, 16)
    histogram = Output(Stream[bytes])

    def build(self):
        print(self.bins)
        return {
            self.histogram: OpLeaf(lambda frame: self.db.ops.Histogram(frame=frame),
                                   [self.frame])
        }

class ScannerColumnSink(OpGraph):
    column = Input(Stream[Any])
    sink = Input(Sink[Any])

    def build(self, column: Stream[bytes]):
        return {
            self.sink: OpLeaf(lambda column: self.db.sinks.Column(columns={'column': column}),
                              [self.column])
        }

class HistogramPipeline(OpGraph):
    video = Output(Stream[FrameType])
    frames = Output(Stream[FrameType])
    histogram = Output(Stream[bytes])
    table = Output(Sink[bytes])

    def build(self, test: int = 1):
        frame = GatheredFrame()
        histogram = Histogram(frame=frame.stream)
        sink = ScannerColumnSink(column=histogram.histogram)
        return {
            self.video: frame.frame,
            self.frames: frame.stream,
            self.histogram: histogram.histogram,
            self.table: sink.column
        }

from .prelude import sample_video
with sample_video(delete=False) as video:
    db = scannerpy.Database()
    p = HistogramPipeline()
    p.run(
        db,
        outputs=[p.table],
        per_video=[{p.video: video, p.table: "test"}],
        globals={p.histogram.bins: 8})
