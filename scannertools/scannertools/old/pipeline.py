from attr import attrs, attrib, evolve
from .video import Video
from collections import namedtuple, defaultdict
import scannerpy
import inspect

class BlockMeta(type):
    def __init__(cls, name, bases, ns):
        super(BlockMeta, cls).__init__(name, bases, ns)
        cls.Output = namedtuple('Output', cls.outputs)

class Block(metaclass=BlockMeta):
    outputs = []
    varying = []

    def _pipeline_initialize(self, db, device):
        self.db = db
        self.device = device

    def validate(self):
        pass

    def fetch_resources(self):
        pass

    def args(self):
        return None

    def build(self, *args, **kwargs):
        raise NotImplemented

    def get_inputs(self, fn):
        sig = inspect.signature(fn)
        return list(sig.parameters)

    def __str__(self):
        return type(self).__name__


class DataSource:
    def load(self):
        raise NotImplemented


class ColumnDataSource(DataSource):
    def __init__(self, column, parser):
        self._column = column
        self._parser = parser

    def load(self):
        return self._column.load(self._parser)

class FrameDataSource(DataSource):
    def __init__(self, video):
        self.video = video

    def load(self):
        raise NotImplemented


class SinkBlock(Block):
    def committed(self, *args, **kwargs):
        raise NotImplemented

class Histogram(Block):
    outputs = ['histogram']

    def build(self, frame):
        return self.Output(histogram=self.db.ops.Histogram(frame=frame))

class FaceDetect(Block):
    outputs = ['face_bboxes']

    def fetch_resources(self):
        try_import('align.detect_face', __name__)
        try_import('tensorflow', __name__)

    def build(self, frame):
        import align, os
        return self.Output(
            face_bboxes=self.db.ops.MTCNNDetectFaces(
                frame=frame,
                model_dir=os.path.dirname(align.__file__),
                device=self.device))

class ColumnSource(Block):
    outputs = ['column']

    def build(self):
        return self.Output(column=self.db.sources.Column())

class FrameSource(Block):
    outputs = ['frame']
    varying = ['video']

    def build(self):
        return self.Output(frame=self.db.sources.FrameColumn())

    def args(self, video):
        return {'frame': self.db.table(video.video.scanner_name()).column('frame')}

class Gather(Block):
    outputs = ['column']
    varying = ['indices']

    def build(self, column):
        return self.Output(column=self.db.streams.Gather(column))

    def args(self, indices):
        return {'column': indices}

class ColumnSink(SinkBlock):
    outputs = ['sink']
    varying = ['table_name']

    def __init__(self, parser):
        self._parser = parser

    def build(self, column):
        return self.Output(sink=self.db.sinks.Column(columns={'column': column}))

    def args(self, table_name):
        return {'sink': table_name}

    def committed(self, table_name):
        return self.db.has_table(table_name) and self.db.table(table_name).committed()

    def output_handles(self, table_name):
        return {'sink': ColumnDataSource(self.db.table(table_name).column('column'), self._parser)}

@attrs(frozen=True)
class VertexAttr:
    vertex = attrib(type=str)
    attribute = attrib(type=str)

    def __str__(self):
        return '{}.{}'.format(self.vertex, self.attribute)

    @classmethod
    def from_str(cls, name):
        parts = name.split('.')

        if len(parts) != 2:
            raise Exception(
                "Attribute path `{}` has an invalid format. Must be of the form `block_name.attribute`.".format(
                    name))

        return cls(vertex=parts[0], attribute=parts[1])

Edge = namedtuple('Edge', ['src', 'dst'])

class BlockGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = [
            Edge(src=VertexAttr.from_str(src), dst=VertexAttr.from_str(dst))
            for (src, dst) in edges
        ]

    def incoming_edges(self, vertex, edges=None):
        edges = edges if edges is not None else self.edges
        return [edge for edge in edges if edge.dst.vertex == vertex]

    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    def toposort(self):
        sorted_list = []

        # Make a copy of edges so we can delete from it
        edges = self.edges[:]

        # Initialize with nodes that have no incoming edges
        to_visit = [k for k in self.nodes.keys()
                    if all([edge.dst.vertex != k for edge in edges])]

        while len(to_visit) > 0:
            parent = to_visit.pop()
            sorted_list.append(parent)

            for child in self.nodes.keys():
                to_child_edges = [i for i, edge in enumerate(edges)
                                  if edge.src.vertex == parent and edge.dst.vertex == child]

                if len(to_child_edges) == 0:
                    continue


                # Remove edges in reverse order to preserve correct indices
                for i in to_child_edges[::-1]:
                    del edges[i]

                if len(self.incoming_edges(child, edges)) == 0:
                    to_visit.append(child)

        if len(edges) > 0:
            raise Exception("Block graph contains a cycle")

        return sorted_list

    def _aligned_str(self, objs, delimiter):
        strs = [(str(x), str(y)) for x, y in objs]
        left_margin = max([len(s) for s, _ in strs])
        return '\n'.join(['{{:>{}}}{{}}{{}}'.format(left_margin).format(x, delimiter, y) for x, y in strs])

    def __str__(self):
        nodes = self._aligned_str(self.nodes.items(), ': ')
        edges = self._aligned_str(self.edges, ' --> ')
        return '=== NODES ===\n{}\n\n=== EDGES ===\n{}'.format(nodes, edges)


class Pipeline:
    def __init__(self, graph):
        self.graph = graph

    def _setup_op_graph(self):
        self.blocks = {}
        for node_name in self.graph.toposort():
            node = self.graph.nodes[node_name]
            required_inputs = node.get_inputs(node.build)
            incoming_edges = self.graph.incoming_edges(node_name)

            kwargs = {}
            for input_name in required_inputs:
                parents = [edge.src for edge in incoming_edges if edge.dst.attribute == input_name]

                if len(parents) == 0:
                    raise Exception("Node `{}` required argument `{}` does not have an input".format(
                        node_name, input_name))

                if len(parents) > 1:
                    raise Exception("Node `{}` required argument `{}` has too many inputs: {}".format(
                        node_name, input_name, ', '.join(['`{}`'.format(p) for p in parents])))

                parent = parents[0]
                parent_block = self.blocks[parent.vertex]

                if not hasattr(parent_block, parent.attribute):
                    raise Exception("Node `{}` does not have requested attribute `{}`".format(
                        parent.vertex, parent.attribute))

                kwargs[input_name] = getattr(parent_block, parent.attribute)

            self.blocks[node_name] = node.build(**kwargs)

    def _ingest_videos(self):
        to_ingest = []
        for job_args in self.per_video:
            for k, v in job_args.items():
                vattr = VertexAttr.from_str(k)
                if isinstance(self.graph.nodes[vattr.vertex], FrameSource) and vattr.attribute == 'video':
                    if not self.db.has_table(v.video.scanner_name()):
                        to_ingest.append(v.video)

        if len(to_ingest) > 0:
            db.ingest_videos([(v.scanner_name(), v.path()) for v in to_ingest], force=True)

    def _build_jobs(self):
        self.jobs = []
        self.job_kwargs = defaultdict(list)
        sorted_nodes = self.graph.toposort()
        for job_args in self.per_video:

            # Convert flattened job args into nested dictionaries
            all_args = defaultdict(dict)
            for k, v in job_args.items():
                vattr = VertexAttr.from_str(k)
                all_args[vattr.vertex][vattr.attribute] = v

            # Build op_args for each node
            op_args = {}
            all_kwargs = {}
           for node_name in sorted_nodes:
                node = self.graph.nodes[node_name]
                kwargs = {}
                for param_name in node.varying:
                    if not node_name in all_args or not param_name in all_args[node_name]:
                        raise Exception("Node `{}` is missing required varying parameter `{}`".format(
                            node_name, param_name))

                    param_value = all_args[node_name][param_name]
                    kwargs[param_name] = param_value

                self.job_kwargs[node_name].append(kwargs)
                block_op_args = node.args(**kwargs)

                # By default, if None is returned, no per-job args to assign
                if block_op_args is None:
                    continue

                for op_name, val in block_op_args.items():
                    op_args[getattr(self.blocks[node_name], op_name)] = val

            self.jobs.append(scannerpy.Job(op_args=op_args))

    def run(self, db, per_video, cache=True, device=scannerpy.DeviceType.CPU):
        self.db = db
        self.per_video = per_video

        for node in self.graph.nodes.values():
            node._pipeline_initialize(db, device)
            node.validate()

        self._setup_op_graph()

        self._ingest_videos()

        self._build_jobs()

        # Find sink op
        sink_blocks = [(name, node) for name, node in self.graph.nodes.items()
                       if all([edge.src.vertex != name for edge in self.graph.edges])]

        if len(sink_blocks) > 1 or len(sink_blocks[0][1].outputs) > 1:
            raise Exception("Scanner does not currently support more than one sink block.")

        block_name, block = sink_blocks[0]
        sink_op = getattr(self.blocks[block_name], block.outputs[0])

        # Filter jobs that aren't already committed
        if cache:
            jobs_to_run = [j for j in self.jobs
                    if not self.graph.nodes[block_name].committed(j.op_args()[sink_op])]
        else:
            jobs_to_run = self.jobs

        # Run Scanner jobs
        if len(jobs_to_run) > 0:
            db.run(sink_op, jobs_to_run, force=True)

        # Build output handles
        all_handles = defaultdict(list)
        for name, node in sink_blocks:
            for kwargs in self.job_kwargs[name]:
                output_handles = node.output_handles(**kwargs)
                for k, v in output_handles.items():
                    all_handles[k].append(v)

        return dict(all_handles)


def run(db, sinks, per_video, **kwargs):
    pipeline = Pipeline(sinks)
    return pipeline.run(db, per_video=per_video, **kwargs)
