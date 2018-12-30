from .prelude import Pipeline, try_import
from scannerpy import Kernel, FrameType, DeviceType
from scannerpy.stdlib.util import download_temp_file
from scannerpy.stdlib import readers
import scannerpy
import pickle
import sys
import os
import numpy as np
from .clothing_detection import TorchKernel
from typing import Sequence

MODEL_URL = 'https://storage.googleapis.com/esper/models/clothing/model_hairstyle.tar'
MODEL_DEF_URL = 'https://raw.githubusercontent.com/Haotianz94/video-analysis/master/streetstyle-classifier/classifier/model_hairstyle.py'


ATTRIBUTES = [
    {
        'key': 'Hair color 3',
        'values': {'black' : 0, 'white': 1, 'blond': 2}, 
    },
    {
        'key': 'Hair color 5',
        'values': {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4}, 
    },
    {
        'key': 'Hair length',
        'values': {'long' : 0, 'medium' : 1, 'short' : 2, 'bald' : 3}
    }
] # yapf: disable


class HairStyle:
    def __init__(self, predictions):
        self._predictions = predictions

    def to_dict(self):
        return {
            attribute['key']: {v: k for k, v in attribute['values'].items()}[prediction]
            for prediction, attribute in zip(self._predictions, ATTRIBUTES)
        }

    def __str__(self):
        pieces = []
        for prediction, attribute in zip(self._predictions, ATTRIBUTES):
            reverse_map = {v: k for k, v in attribute['values'].items()}
            pieces.append('{}: {}'.format(attribute['key'], reverse_map[prediction]))
        return '\n'.join(pieces)

BATCH_SIZE = 10

@scannerpy.register_python_op(name='DetectHairStyleCPU', device_type=DeviceType.CPU, batch=BATCH_SIZE)
@scannerpy.register_python_op(name='DetectHairStyleGPU', device_type=DeviceType.GPU, batch=BATCH_SIZE)
class DetectHairStyle(TorchKernel):
    def __init__(self, config):
        from torchvision import transforms
        TorchKernel.__init__(self, config)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def execute(self, frame: Sequence[FrameType], bboxes: Sequence[bytes]) -> Sequence[bytes]:
        from PIL import Image
        from torch.autograd import Variable
        import torch

        h, w = frame[0].shape[:2]

        counts = []
        images = []
        for i, (fram, bbs) in enumerate(zip(frame, bboxes)):
            bbs = readers.bboxes(bbs, self.config.protobufs)
            counts.append((counts[i - 1][0] + counts[i - 1][1] if i > 0 else 0, len(bbs)))
            if len(bboxes) == 0:
                raise Exception("No bounding boxes")

            # for bbox in bbs:
            #     # print(int(bbox.y1 * h), int(bbox.y2 * h),
            #     #       int(bbox.x1 * w), int(bbox.x2 * w))
            #     Image.fromarray(fram[int(bbox.y1 * h):int(bbox.y2 * h),
            #                          int(bbox.x1 * w):int(bbox.x2 * w)])

            images.extend([
                fram[int(bbox.y1 * h):int(bbox.y2 * h),
                     int(bbox.x1 * w):int(bbox.x2 * w)] for bbox in bbs
            ])

        all_scores = []
        for i in range(0, len(images), BATCH_SIZE):
            tensor = self.images_to_tensor([self.transform(Image.fromarray(img)) for img in images[i:i+BATCH_SIZE]])
            var = Variable(tensor if self.cpu_only else tensor.cuda(), requires_grad=False)
            all_scores.append(self.model(var))

        scores = [
            torch.cat([scores[i] for scores in all_scores], dim=0)
            for i in range(len(all_scores[0]))
        ]

        all_att = []
        for k in range(len(frame)):
            (idx, n) = counts[k]
            predicted_attributes = np.zeros((n, len(scores)), dtype=np.int32)
            for i, attrib_score in enumerate(scores):
                _, predicted = torch.max(attrib_score[idx:idx+n, :], 1)
                predicted_attributes[:, i] = predicted.cpu().data.numpy().astype(np.int32)
            all_att.append(pickle.dumps(predicted_attributes))

        return all_att


def parse_hairstyle(s, _proto):
    predictions = pickle.loads(s)
    return [HairStyle(predictions[i, :]) for i in range(len(predictions))]


class HairStyleDetectionPipeline(Pipeline):
    job_suffix = 'hairstyle'
    parser_fn = lambda: parse_hairstyle
    additional_sources = ['bboxes']

    def fetch_resources(self):
        try_import('torch', __name__)
        try_import('torchvision', __name__)

        self._model_path = download_temp_file(MODEL_URL)
        self._model_def_path = download_temp_file(MODEL_DEF_URL)

    def build_pipeline(self, adjust_bboxes=True):
        return {
            'hairstyle':
            self._db.ops.DetectHairStyle(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op,
                model_path=self._model_path,
                model_def_path=self._model_def_path,
                model_key='best_model',
                adjust_bboxes=adjust_bboxes,
                device=self._device)
        }


detect_hairstyle = HairStyleDetectionPipeline.make_runner()
