from scannertools import optical_flow, vis, sample_video
import scannerpy
import os

with sample_video() as video:
    db = scannerpy.Database()
    flow_fields = optical_flow.compute_flow(db, videos=[video], frames=[[0]])
    vis.draw_flows(db, videos=[video], frames=[[0]], flows=flow_fields, paths=['sample_flow.mp4'])
    print('Wrote video with flow drawn to {}'.format(os.path.abspath('sample_flow.mp4')))
