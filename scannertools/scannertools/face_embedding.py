from scannerpy import FrameType, DeviceType
import scannerpy
from scannerpy.util import download_temp_file, temp_directory
from .tensorflow import TensorFlowKernel
import os
import numpy as np
from scannerpy.types import BboxList

MODEL_FILE = 'https://storage.googleapis.com/esper/models/facenet/20170512-110547.tar.gz'


@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]])
class EmbedFaces(TensorFlowKernel):
    def build_graph(self):
        import tensorflow as tf
        self.images_placeholder = None
        self.g = tf.Graph()
        self._g_default = self.g.as_default()
        self._model_dir = os.path.join(temp_directory(), '20170512-110547')
        return self.g

    def fetch_resources(self):
        download_temp_file(MODEL_FILE, untar=True)

    def execute(self, frame: FrameType, bboxes: BboxList) -> bytes:
        import facenet
        import cv2
        import tensorflow as tf

        if self.images_placeholder is None:
            print('Loading model...')
            with self.g.as_default():
                with self.sess.as_default():
                    model_path = self._model_dir
                    meta_file, ckpt_file = facenet.get_model_filenames(model_path)
                    saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))
                    saver.restore(self.sess, os.path.join(model_path, ckpt_file))

                    self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
                    self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                        'phase_train:0')
            print('Model loaded!')

        [h, w] = frame.shape[:2]

        out_size = 160
        outputs = b''
        for bbox in bboxes:
            # NOTE: if using output of mtcnn, not-normalized, so removing de-normalization factors here
            face_img = frame[int(bbox.y1 * h):int(bbox.y2 * h), int(bbox.x1 * w):int(bbox.x2 * w)]
            [fh, fw] = face_img.shape[:2]
            if fh == 0 or fw == 0:
                outputs += np.zeros(128, dtype=np.float32).tobytes()
            else:
                face_img = cv2.resize(face_img, (out_size, out_size))
                face_img = facenet.prewhiten(face_img)
                embs = self.sess.run(
                    self.embeddings,
                    feed_dict={
                        self.images_placeholder: [face_img],
                        self.phase_train_placeholder: False
                    })

                outputs += embs[0].tobytes()

        return b'\0' if outputs == b'' else outputs
