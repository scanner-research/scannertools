from .prelude import Pipeline, try_import
from scannerpy import Kernel, FrameType, DeviceType
from scannerpy.stdlib.util import download_temp_file
from scannerpy.stdlib import readers
import scannerpy
import pickle
import sys
import os
import numpy as np

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

    def __str__(self):
        pieces = []
        for prediction, attribute in zip(self._predictions, ATTRIBUTES):
            reverse_map = {v: k for k, v in attribute['values'].items()}
            pieces.append('{}: {}'.format(attribute['key'], reverse_map[prediction]))
        return '\n'.join(pieces)


class TorchKernel(Kernel):
    def __init__(self, config):
        import torch

        self.config = config

        self.cpu_only = True
        visible_device_list = []
        for handle in config.devices:
            if handle.type == DeviceType.GPU.value:
                visible_device_list.append(handle.id)
                self.cpu_only = False

        self.model = self.build_model()

        if not self.cpu_only:
            torch.cuda.set_device(visible_device_list[0])
            self.model = self.model.cuda()

        # Not sure if this is necessary? Haotian had it in his code
        self.model.eval()

    def images_to_tensor(self, images):
        import torch

        shape = images[0].shape
        images_tensor = torch.Tensor(len(images), shape[0], shape[1], shape[2])
        for i in range(len(images)):
            images_tensor[i] = images[i]
        return images_tensor

    def build_model(self):
        import torch

        sys.path.insert(0, os.path.split(self.config.args['model_def_path'])[0])
        kwargs = {'map_location': lambda storage, location: storage} if self.cpu_only else {}
        return torch.load(self.config.args['model_path'], **kwargs)[self.config.args['model_key']]

    def close(self):
        del self.model

    def execute(self):
        raise NotImplementedError


@scannerpy.register_python_op()
class DetectHairStyle(TorchKernel):
    def __init__(self, config):
        from torchvision import transforms
        TorchKernel.__init__(self, config)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def execute(self, frame: FrameType, bboxes: bytes) -> bytes:
        from PIL import Image
        from torch.autograd import Variable
        import torch

        H, W = frame.shape[:2]
        bboxes = readers.bboxes(bboxes, self.config.protobufs)
        if len(bboxes) == 0:
            return pickle.dumps([])

        if self.config.args['adjust_bboxes']:
            images = []
            for i, bbox in enumerate(bboxes):
                x1 = int(bbox.x1 * W)
                y1 = int(bbox.y1 * H)
                x2 = int(bbox.x2 * W)
                y2 = int(bbox.y2 * H)
                w = max(y2 - y1, x2 - x1) * 3 // 4
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                x1 = cx - w if cx - w > 0 else 0
                x2 = cx + w if cx + w < W else W
                y1 = cy - w if cy - w > 0 else 0
                y2 = cy + w if cy + w < H else H
                cropped = frame[y1:y2, x1:x2, :]
                images.append(cropped)
        else:
            images = [
                frame[int(bbox.y1 * h):int(bbox.y2 * h),
                      int(bbox.x1 * w):int(bbox.x2 * w)] for bbox in bboxes
            ]

        tensor = self.images_to_tensor([self.transform(Image.fromarray(img)) for img in images])
        var = Variable(tensor if self.cpu_only else tensor.cuda(), requires_grad=False)
        scores, features = self.model(var)

        predicted_attributes = np.zeros((len(images), len(scores)), dtype=np.int32)
        for i, attrib_score in enumerate(scores):
            _, predicted = torch.max(attrib_score, 1)
            predicted_attributes[:, i] = predicted.cpu().data.numpy().astype(np.int32)

        return pickle.dumps(predicted_attributes)


def parse_hairstyle(s, _proto):
    predictions = pickle.loads(s)
    return [HairStyle(predictions[i, :]) for i in range(len(predictions))]


class HairStyleDetectionPipeline(Pipeline):
    job_suffix = 'hairstyle'
    parser_fn = lambda _: parse_hairstyle
    additional_sources = ['bboxes']
    run_opts = {'pipeline_instances_per_node': 1}

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
                adjust_bboxes=adjust_bboxes)
        }


detect_hairstyle = HairStyleDetectionPipeline.make_runner()
