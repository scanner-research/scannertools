from attr import attrs, attrib, evolve
from .video import Video
from collections import namedtuple, defaultdict
import scannerpy
import inspect
from scannerpy import FrameType
from typing import Generic, TypeVar, Any

T = TypeVar('T')

class Stream(Generic[T]):
    pass

class OpGraphMeta(type):
    def __init__(cls, name, bases, ns):
        super(OpGraphMeta, cls).__init__(name, bases, ns)
        cls.Aliases = namedtuple('Aliases', cls.aliases)

@attrs(frozen=True)
class Node:
    graph = attrib()
    name = attrib()

    def get(self):
        return getattr(self.graph._aliases, self.name)

    def resolve(self):
        next = self.get()
        if isinstance(next, Node):
            return next.resolve()
        else:
            assert(isinstance(next, OpLeaf))
            return next._op

    def __getattr__(self, name):
        return Node(graph=self, name=name)

class OpGraph(metaclass=OpGraphMeta):
    aliases = []

    def __init__(self, *args, **kwargs):
        self._build_args = (args, kwargs)
        self._built = False

    def __getattr__(self, name):
        build_params = list(inspect.signature(self.build).parameters.keys())
        if name in self.aliases or name in build_params:
            return Node(graph=self, name=name)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

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


        my_globals = {k.name: v for k, v in globals.items() if k.graph == self}
        aliases = self.build(*args, **kwargs, **my_globals)
        self._aliases = aliases

        #nodes = self._search_and_toposort(aliases)
        nodes = list(aliases._asdict().values())

        for node in nodes:
            if isinstance(node, OpLeaf):
                graph = node
            elif not isinstance(node, Node):
                raise Exception("Object {} is not a Node".format(node))
            else:
                graph = node.graph

            graph.initialize(db)
            graph._build({
                Node(k.graph.get(), k.name): v
                for k, v in globals.items() if isinstance(k.graph, Node) and k.graph.get() == node
             })

        return type(aliases)(**{k: v.get() if isinstance(v, Node) else v for k, v in aliases._asdict().items()})

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
            elif not isinstance(node, Node):
                raise Exception("Object {} is not a Node".format(node))
            else:
                graph = node.graph

            graph.initialize(db)
            graph._build({})

        op = self._f(*[node.get() if isinstance(node, Node) else node._op for node in self._args])
        self._op = op
        return op

class ScannerFrameSource(OpGraph):
    aliases = ['frame']

    def build(self):
        return self.Aliases(frame=OpLeaf(lambda: self.db.sources.FrameColumn()))

class Gather(OpGraph):
    aliases = ['stream']

    def build(self, stream: Stream[Any]):
        return self.Aliases(stream=OpLeaf(lambda stream: self.db.streams.Gather(stream), [stream]))

class GatheredFrame(OpGraph):
    aliases = ['frame', 'stream']

    def build(self):
        frame = ScannerFrameSource()
        gather = Gather(stream=frame.frame)
        return self.Aliases(frame=frame.frame, stream=gather.stream)

class Histogram(OpGraph):
    aliases = ['histogram']

    def build(self, frame: Stream[FrameType], bins: int = 16):
        return self.Aliases(histogram=OpLeaf(lambda frame: self.db.ops.Histogram(frame=frame), [frame]))

class ScannerColumnSink(OpGraph):
    aliases = ['column']

    def build(self, column: Stream[bytes]):
        return self.Aliases(
            column=OpLeaf(lambda column: self.db.sinks.Column(columns={'column': column}), [column]))

class HistogramPipeline(OpGraph):
    aliases = ['video', 'frames', 'histogram', 'table']

    def build(self, test: int = 1):
        frame = GatheredFrame()
        histogram = Histogram(frame=frame.stream)
        sink = ScannerColumnSink(column=histogram.histogram)
        return self.Aliases(
            video=frame.frame,
            frames=frame.stream,
            histogram=histogram.histogram,
            table=sink.column)

from .prelude import sample_video
with sample_video(delete=False) as video:
    db = scannerpy.Database()
    p = HistogramPipeline()
    p.run(db,
          outputs=[p.table],
          per_video=[{p.video: video, p.table: "test"}],
          globals={p.histogram.bins: 8, p.test: 2})
