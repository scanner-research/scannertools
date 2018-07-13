from .prelude import Pipeline, make_pipeline_runner, try_import
from scannerpy import FrameType, Kernel
import scannerpy
from scannerpy.stdlib.util import download_temp_file
from scannerpy.stdlib import readers
import os
import numpy as np

MODEL_FILE = 'https://storage.googleapis.com/esper/models/facenet/20170512-110547.tar.gz'


@scannerpy.register_python_op()
class EmbedFaces(Kernel):
    def __init__(self, config):
        import tensorflow as tf
        import facenet
        self.config = config
        self.sess = tf.Session()
        model_path = config.args['model_dir']
        meta_file, ckpt_file = facenet.get_model_filenames(model_path)
        g = tf.Graph()
        g.as_default()
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))
        saver.restore(self.sess, os.path.join(model_path, ckpt_file))

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

    def close(self):
        self.sess.close()

    def execute(self, frame: FrameType, bboxes: bytes) -> bytes:
        import facenet
        import cv2
        [h, w] = frame.shape[:2]

        out_size = 160
        bboxes = readers.bboxes(bboxes, self.config.protobufs)
        outputs = b''
        for bbox in bboxes:
            # NOTE: if using output of mtcnn, not-normalized, so removing de-normalization factors here
            face_img = frame[int(bbox.y1*h):int(bbox.y2*h), int(bbox.x1*w):int(bbox.x2*w)]
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

        return ' ' if outputs == b'' else outputs


class FaceEmbeddingPipeline(Pipeline):
    job_suffix = 'embed'
    parser_fn = lambda _: readers.array(np.float32)
    run_opts = {'pipeline_instances_per_node': 1}
    required_sources = ['videos', 'bboxes']

    def fetch_resources(self):
        try_import('facenet', __name__)
        try_import('tensorflow', __name__)
        self._model_dir = download_temp_file(MODEL_FILE, untar=True) + '/20170512-110547'

    def build_pipeline(self):
        return {
            'embeddings':
            self._db.ops.EmbedFaces(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op,
                model_dir=self._model_dir)
        }


embed_faces = make_pipeline_runner(FaceEmbeddingPipeline)
