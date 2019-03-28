import scannerpy

def frame_gather_graph(db, video, frame_indices):
    frame = db.sources.FrameColumn(video=video)
    frame_gathered = db.streams.Gather(frame, indices=frame_indices)
    return frame_gathered

def default_graph(kernel, sink=None):
    if not sink:
        sink = db.sinks.Column(columnsd={'column': output}, table_name=table_name)
    def graph(db, video, frame_indices, table_name):
        frame_gathered = frame_gather_graph(db, video, frame_indices)
        output = kernel(db, frame_gathered)
        sink = db.sinks.Column(columns={'column': output}, table_name=table_name)
        return sink
    return graph


histogram_graph = default_graph(lambda db, frame: db.ops.Histogram(frame=frame))

from .prelude import sample_video
with sample_video(delete=False) as video:
    db = scannerpy.Database()
    print(histogram_graph(db, video=[video], frame_indices=[[0]], table_name=['test']))
