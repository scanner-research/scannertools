ARG tag
FROM scannerresearch/scanner:${tag}-latest
ARG tag2

WORKDIR /opt

# Fixes travis pip failure
RUN rm /usr/share/python-wheels/urllib3-1.13.1-py2.py3-none-any.whl && \
    rm /usr/share/python-wheels/requests-2.9.1-py2.py3-none-any.whl && \
    pip3 install requests[security] --upgrade -v
RUN pip3 install face-alignment scipy pysrt
RUN pip3 install --upgrade setuptools
RUN if [ "$tag2" = "cpu" ]; then pip3 install tensorflow==1.12.0; else pip3 install tensorflow-gpu==1.12.0; fi
RUN git clone https://github.com/scanner-research/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH

# Install Pytorch 1.0 (Caffe2 included)
RUN pip3 install torchvision_nightly
RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# Install maskrcnn dependencies
RUN git clone https://www.github.com/nvidia/apex && \
    cd apex && pip3 install . && cd .. && rm -rf apex
RUN pip3 install yacs

# Install maskrcnn-benchmark
ARG force_cuda
ENV FORCE_CUDA=${force_cuda}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python3 setup.py build develop
ENV PYTHONPATH /opt/maskrcnn-benchmark:$PYTHONPATH

# Install cocoapi
RUN git clone https://github.com/scanner-research/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && pip3 install Cython \
 && python3 setup.py build_ext install \
 && rm -rf build

# Install DensePose
RUN git clone -b python3 https://github.com/scanner-research/DensePose.git \
 && cd DensePose \
 && pip3 install -r requirements.txt \
 && make \
 && cd DensePoseData && bash get_densepose_uv.sh
ENV PYTHONPATH /opt/DensePose:$PYTHONPATH

RUN apt-get update && apt-get install -y jq

RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk kubectl

# https://github.com/keras-team/keras/issues/9567#issuecomment-370887563
RUN if [ "$tag2" != "cpu" ]; then \
    apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
            libcudnn7=7.0.5.15-1+cuda9.0 \
            libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
            rm -rf /var/lib/apt/lists/*; \
    fi

COPY . scannertools
RUN cd scannertools && pip3 install --upgrade setuptools && ./scripts/install-all.sh

WORKDIR /app