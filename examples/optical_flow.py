import tixelbox as tb
import tixelbox.optical_flow as optflow
import os

with tb.sample_video(delete=False) as video:
    flow_fields = optflow.compute_flow(video)
    optflow.draw_flow_fields(video, flow_fields, path='flow.mp4')
