import scannerpy as sp
from scannertools.torch import TorchKernel
import torchvision.models as models
import torchvision.transforms as transforms
from scannerpy.types import NumpyArrayFloat32
from typing import Sequence
import torch
import numpy as np
from timeit import default_timer as now

@sp.register_python_op(batch=2, device_sets=[(sp.DeviceType.CPU, 0), (sp.DeviceType.GPU, 1)])
class Resnet(TorchKernel):
    def build_model(self):
        self._mu = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        self._sigma = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        if not self.cpu_only:
            self._mu = self._mu.cuda()
            self._sigma = self._sigma.cuda()

        return models.resnet18(pretrained=True)

    def execute(self, frame: Sequence[sp.FrameType]) -> Sequence[NumpyArrayFloat32]:
        batch_size = len(frame)

        start = now()
        batch_tensor = torch.from_numpy(
            np.moveaxis(np.concatenate(np.expand_dims(frame, axis=0), axis=0), 3, 1)) \
            .type(torch.FloatTensor)

        if not self.cpu_only:
            start = now()
            batch_tensor = batch_tensor.cuda()
            #print('Transfer to device: {:.3f}'.format(now() - start))

        batch_tensor /= 255.0

        batch_tensor -= self._mu
        batch_tensor /= self._sigma
        #print('Transform: {:.3f}'.format(now() - start))

        with torch.no_grad():
            start = now()
            output = self.model.forward(batch_tensor)
            #print('Forward: {:.3f}'.format(now() - start))

        if not self.cpu_only:
            start = now()
            output = output.cpu()
            #print('Transfer from device: {:.3f}'.format(now() - start))

        import sys
        sys.stdout.flush()

        return [output[i, :].numpy() for i in range(batch_size)]
