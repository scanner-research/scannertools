from scannerpy.util import download_temp_file, temp_directory
from scannerpy.types import BboxList
from scannerpy import FrameType
from typing import Any
import scannerpy
import cv2
import pickle
import os

MODEL_FILE = 'https://storage.googleapis.com/esper/models/rude-carnie/21936.tar.gz'


@scannerpy.register_python_op()
class DetectGender(scannerpy.Kernel):
    def fetch_resources(self):
        download_temp_file(MODEL_FILE, untar=True)

    def setup_with_resources(self):
        from carnie_helper import RudeCarnie
        self.rc = RudeCarnie(model_dir=os.path.join(temp_directory(), '21936'))

    def execute(self, frame: FrameType, bboxes: BboxList) -> Any:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        [h, w] = frame.shape[:2]
        frames = [
            frame[int(bbox.y1 * h):int(bbox.y2 * h),
                  int(bbox.x1 * w):int(bbox.x2 * w)] for bbox in bboxes
        ]
        genders = self.rc.get_gender_batch(frames)
        return genders
