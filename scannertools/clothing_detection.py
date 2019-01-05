from .prelude import Pipeline, try_import
from scannerpy import Kernel, FrameType, DeviceType
from scannerpy.stdlib.util import download_temp_file
from scannerpy.stdlib import readers, writers
import scannerpy
import pickle
import sys
import os
import numpy as np
from typing import Sequence

MODEL_URL = 'https://storage.googleapis.com/esper/models/clothing/model_newsanchor.tar'
MODEL_DEF_URL = 'https://raw.githubusercontent.com/Haotianz94/video-analysis/master/streetstyle-classifier/classifier/newsanchor_classifier_model.py'


ATTRIBUTES = [
    {
        'key': 'Clothing pattern',
        'values': {'solid': 0, 'graphics' : 1, 'striped' : 2, 'floral' : 3, 'plaid' : 4, 'spotted' : 5}, # clothing_pattern
    },
    {
        'key': 'Major color',
        'values': {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
            'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
            'cyan' : 12, 'dark blue' : 13}, # major_color
    },
    {
        'key': 'Wearing necktie',
        'values': {'necktie no': 0, 'necktie yes' : 1}, # wearing_necktie
    },
    {
        'key': 'Collar presence',
        'values': {'collar no': 0, 'collar yes' : 1}, # collar_presence
    },
    {
        'key': 'Wearing scarf',
        'values': {'scarf no': 0, 'scarf yes' : 1}, # wearing_scarf
    },
    {
        'key': 'Sleeve length',
        'values': {'long sleeve' : 0, 'short sleeve' : 1, 'no sleeve' : 2}, # sleeve_length
    },
    {
        'key': 'Neckline shape',
        'values': {'round' : 0, 'folded' : 1, 'v-shape' : 2}, # neckline_shape
    },
    {
        'key': 'Clothing category',
        'values': {'shirt' : 0, 'outerwear' : 1, 't-shirt' : 2, 'dress' : 3,
                   'tank top' : 4, 'suit' : 5, 'sweater' : 6}, # clothing_category
    },
    {
        'key': 'Wearing jacket?',
        'values': {'jacket no': 0, 'jacket yes' : 1}, # wearing_jacket
    },
    {
        'key': 'Wearing hat?',
        'values': {'hat no': 0, 'hat yes' : 1}, # wearing_hat
    },
    {
        'key': 'Wearing glasses?',
        'values': {'glasses no': 0, 'glasses yes' : 1}, # wearing_glasses
    },
    {
        'key': 'Multiple layers?',
        'values': {'one layer': 0, 'more layer' : 1}, # multiple_layers
    },
    {
        'key': 'Necktie color',
        'values': {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
            'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
            'cyan' : 12, 'dark blue' : 13}, # necktie_color
    },
    {
        'key': 'Necktie pattern',
        'values': {'solid' : 0, 'striped' : 1, 'spotted' : 2}, # necktie_pattern
    },
    {
        'key': 'Hair color',
        'values': {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4}, # hair_color
    },
    {
        'key': 'Hair length',
        'values': {'long' : 0, 'medium' : 1, 'short' : 2, 'bald' : 3} # hair_length
    }
] # yapf: disable


class Clothing:
    def __init__(self, predictions):
        self._predictions = predictions

    def to_dict(self):
        return {
            attribute['key']: {v: k for k, v in attribute['values'].items()}[prediction]
            for prediction, attribute in zip(self._predictions, ATTRIBUTES)
        }

    def __str__(self):
        return '\n'.join([
            '{}: {}'.format(k, v)
            for k, v in self.to_dict().items()
        ])


class TorchKernel(Kernel):
    def __init__(self, config):
        import torch

        self.config = config

        self.cpu_only = True
        visible_device_list = []
        for handle in config.devices:
            if int(handle.type) == DeviceType.GPU.value:
                visible_device_list.append(handle.id)
                self.cpu_only = False

        self.model = self.build_model()
        
        if not self.cpu_only:
            print('Using GPU: {}'.format(visible_device_list[0]))
            torch.cuda.set_device(visible_device_list[0])
            self.model = self.model.cuda()
        else:
            print('Using CPU')

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


@scannerpy.register_python_op(batch=20)
class PrepareClothingBbox(Kernel):

    def detect_edge_text(self, img, start_y=40):
        import cv2
        edges = cv2.Canny(img, 80, 80)
        img_bright = np.max(img, axis=2)
        H, W = img_bright.shape
        img_bright = img_bright.astype('int')
        BOUNDARY_THRESH = 0.5
        CONTRAST_THRESH = 96
        TEXT_THRESH = 0.45
        HEAD_THRESH = 0.3
        start_y = int((H - start_y) * HEAD_THRESH + start_y)
        for y in range(start_y, H):
            ## detect edge
            find_edge = False
            find_text = False
            non_zero = np.count_nonzero(edges[y])
            if 1. * non_zero / W > BOUNDARY_THRESH:
                find_edge = True
            ## find text
            cnt_grad = 0
            for x in range(W):
                grad_horiz = False
                neighbor = [-2, -1, 1, 2]
                for i in neighbor:
                    if x+i < 0 or x+i > W-1:
                        continue
                    if np.fabs(img_bright[y, x+i] - img_bright[y, x]) > CONTRAST_THRESH:
                        grad_horiz = True
                        break
                if grad_horiz:
                    cnt_grad += 1
            if 1. * cnt_grad / W > TEXT_THRESH:
                find_text = True
            if find_edge or find_text:
                ## too close to head: possibly wearing texture
                return y
        return H

    def execute(self, frame: Sequence[FrameType], bboxes: Sequence[bytes]) -> Sequence[bytes]:
        h, w = frame[0].shape[:2]
        all_new_bbs = []
        for fram, bbs in zip(frame, bboxes):
            bbs = readers.bboxes(bbs, self.protobufs)
            new_bbs = []
            for i, bbox in enumerate(bbs):
                x1 = int(bbox.x1 * w)
                y1 = int(bbox.y1 * h)
                x2 = int(bbox.x2 * w)
                y2 = int(bbox.y2 * h)

                ## set crop window
                crop_w = (x2 - x1) * 2
                crop_h = crop_w * 2
                X1 = int((x1 + x2) / 2 - crop_w / 2)
                X2 = X1 + crop_w
                Y1 = int((y1 + y2) / 2 - crop_h / 3)
                Y2 = Y1 + crop_h

                ## adjust box size by image boundary
                crop_x1 = max(0, X1)
                crop_x2 = min(w-1, X2)
                crop_y1 = max(0, Y1)
                crop_y2 = min(h-1, Y2)
                cropped = fram[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]

                ## compute body bound
                body_bound = 1.0
                for j, other_bbox in enumerate(bbs):
                    if i == j:
                        continue
                    if bbox.y2 < other_bbox.y1:
                        center = (bbox.x1  + bbox.x2) / 2
                        crop_x1 = (center - bbox.x2 + bbox.x1)
                        crop_x2 = (center + bbox.x2 - bbox.x1)
                        if other_bbox.x1 < crop_x2 or other_bbox.x2 > crop_x1:
                            body_bound = other_bbox.y1

                ## detect edge and text
                neck_line = y2 - crop_y1
                body_bound = int(body_bound * h) - crop_y1
                crop_y = self.detect_edge_text(cropped, neck_line)
                crop_y = min(crop_y, body_bound)

                def inbound(coord, limit):
                    return 0 <= int(coord) and int(coord) < limit

                # If we produce a malformed bbox then just use the original bbox.
                if abs(crop_x1 - crop_x2) < 20 or abs(crop_y1 - crop_y) < 20 \
                   or crop_x1 >= crop_x2 or crop_y1 >= crop_y \
                   or not inbound(crop_x1, w) or not inbound(crop_x2, w) \
                   or not inbound(crop_y1, h) or not inbound(crop_y, h):
                    new_bbs.append(bbox)
                else:
                    new_bbs.append(self.protobufs.BoundingBox(
                        x1=crop_x1/w,
                        x2=crop_x2/w,
                        y1=crop_y1/h,
                        y2=crop_y/h))

            all_new_bbs.append(writers.bboxes(new_bbs, self.protobufs))
        return all_new_bbs
            

@scannerpy.register_python_op(name='DetectClothingCPU', device_type=DeviceType.CPU, batch=10)
@scannerpy.register_python_op(name='DetectClothingGPU', device_type=DeviceType.GPU, batch=10)
class DetectClothing(TorchKernel):
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

            for bbox in bbs:
                # print(int(bbox.y1 * h), int(bbox.y2 * h),
                #       int(bbox.x1 * w), int(bbox.x2 * w))
                Image.fromarray(fram[int(bbox.y1 * h):int(bbox.y2 * h),
                                     int(bbox.x1 * w):int(bbox.x2 * w)])

            images.extend([
                fram[int(bbox.y1 * h):int(bbox.y2 * h),
                     int(bbox.x1 * w):int(bbox.x2 * w)] for bbox in bbs
            ])


        tensor = self.images_to_tensor([self.transform(Image.fromarray(img)) for img in images])
        var = Variable(tensor if self.cpu_only else tensor.cuda(), requires_grad=False)
        scores, features = self.model(var)

        all_att = []
        for k in range(len(frame)):
            (idx, n) = counts[k]
            predicted_attributes = np.zeros((n, len(scores)), dtype=np.int32)
            for i, attrib_score in enumerate(scores):
                _, predicted = torch.max(attrib_score[idx:idx+n, :], 1)
                predicted_attributes[:, i] = predicted.cpu().data.numpy().astype(np.int32)
            all_att.append(pickle.dumps(predicted_attributes))

        return all_att


def parse_clothing(s, _proto):
    predictions = pickle.loads(s)
    return [Clothing(predictions[i, :]) for i in range(len(predictions))]


class ClothingDetectionPipeline(Pipeline):
    job_suffix = 'clothing'
    parser_fn = lambda: parse_clothing
    additional_sources = ['bboxes']

    def fetch_resources(self):
        try_import('torch', __name__)
        try_import('torchvision', __name__)

        self._model_path = download_temp_file(MODEL_URL)
        self._model_def_path = download_temp_file(MODEL_DEF_URL)

    def build_pipeline(self, adjust_bboxes=True):
        new_bbs = PrepareClothingBbox(
            frame=self._sources['frame_sampled'].op,
            bboxes=self._sources['bboxes'].op)
        return {
            'clothing':
            getattr(self._db.ops, 'DetectClothing{}'.format('GPU' if self._device == DeviceType.GPU else 'CPU'))(
                frame=self._sources['frame_sampled'].op,
                bboxes=new_bbs,
                model_path=self._model_path,
                model_def_path=self._model_def_path,
                model_key='best_model',
                adjust_bboxes=adjust_bboxes,
                device=self._device)
        }


detect_clothing = ClothingDetectionPipeline.make_runner()
