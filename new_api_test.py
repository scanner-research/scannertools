from scannertools.prelude import sample_video
from scannerpy import Database
from scannerpy.stdlib import readers
from scannertools.pipeline import *

if __name__ == "__main__":
    with sample_video(delete=False) as video:
        db = Database()

        graph = BlockGraph(
            nodes = {
                'frame': FrameSource(),
                'frame_sampled': Gather(),
                #'histogram': Histogram(),
                'face_detect': FaceDetect(),
                'sink': ColumnSink(readers.bboxes)
            },
            edges = [
                ('frame.frame', 'frame_sampled.column'),
                ('frame_sampled.column', 'face_detect.frame'),
                ('face_detect.face_bboxes', 'sink.column')
                #('frame_sampled.column', 'histogram.frame'),
                #('histogram.histogram', 'sink.column')
            ]
        )

        HistogramPipeline = Pipeline(graph)

        hists = HistogramPipeline.run(
            db,
            per_video=[{
                'frame.video': FrameDataSource(video),
                'frame_sampled.indices': [0, 1, 2],
                'sink.table_name': 'face'
            }],
            cache=False)

        h = list(hists['sink'][0].load())
        print(h)
