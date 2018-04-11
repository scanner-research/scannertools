import tixelbox as tb
import tixelbox.optical_flow as optflow
import tixelbox.vis as vis
import scannerpy
import os

with tb.WithMany(scannerpy.Database(), tb.sample_video()) as (db, video):
    flow_fields = optflow.compute_flow(db, video)
    vis.draw_flow_fields(db, video, flow_fields, path='sample_flow.mp4')
    print('Wrote video with flow drawn to {}'.format(os.path.abspath('sample_flow.mp4')))
