from scannerpy import DeviceType, FrameType, register_python_op, protobufs
from scannerpy.types import BboxList
from .tensorflow import TensorFlowKernel
from typing import List
import os.path
import numpy as np

@register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]])
class MTCNNDetectFaces(TensorFlowKernel):
    def build_graph(self):
        import tensorflow as tf
        self.pnet = None
        self.g = tf.Graph()
        self._g_default = self.g.as_default()
        return self.g

    def execute(self, frame: FrameType) -> BboxList:
        import align
        import align.detect_face

        if self.pnet is None:
            with self.g.as_default():
                with self.sess.as_default():
                    print('Loading model...')
                    self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(
                        self.sess, os.path.dirname(align.__file__))
                    print('Model loaded!')

        threshold = [0.45, 0.6, 0.7]
        factor = 0.709
        vmargin = 0.2582651235637604
        hmargin = 0.3449094129917718
        out_size = 160
        detection_window_size_ratio = .2

        imgs = [frame]
        #print(('Face detect on {} frames'.format(len(imgs))))
        detections = align.detect_face.bulk_detect_face(
            imgs, detection_window_size_ratio, self.pnet, self.rnet, self.onet, threshold, factor)

        batch_faces = []
        for img, bounding_boxes in zip(imgs, detections):
            if bounding_boxes == None:
                batch_faces.append([])
                continue
            frame_faces = []
            bounding_boxes = bounding_boxes[0]
            num_faces = bounding_boxes.shape[0]
            for i in range(num_faces):
                confidence = bounding_boxes[i][4]
                if confidence < .1:
                    continue

                img_size = np.asarray(img.shape)[0:2]
                det = np.squeeze(bounding_boxes[i][0:5])
                vmargin_pix = int((det[2] - det[0]) * vmargin)
                hmargin_pix = int((det[3] - det[1]) * hmargin)
                frame_faces.append(
                    protobufs.BoundingBox(
                        x1=np.maximum(det[0] - hmargin_pix / 2, 0) / img_size[1],
                        y1=np.maximum(det[1] - vmargin_pix / 2, 0) / img_size[0],
                        x2=np.minimum(det[2] + hmargin_pix / 2, img_size[1]) / img_size[1],
                        y2=np.minimum(det[3] + vmargin_pix / 2, img_size[0]) / img_size[0],
                        score=det[4]))

            batch_faces.append(frame_faces)

        return batch_faces[0]
