ARG tag
FROM scannerresearch/scanner:${tag}-latest
ARG tag2

WORKDIR /opt

# Fixes travis pip failure
RUN rm /usr/share/python-wheels/urllib3-1.13.1-py2.py3-none-any.whl && \
    rm /usr/share/python-wheels/requests-2.9.1-py2.py3-none-any.whl && \
    pip3 install requests[security] --upgrade -v
RUN pip3 install face-alignment scipy pysrt
RUN if [ "$tag2" = "cpu" ]; then pip3 install tensorflow==1.12.0; else pip3 install tensorflow-gpu==1.12.0; fi
RUN git clone https://github.com/scanner-research/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH

# pytorch (specific version for maskRCNN)
RUN pip3 install torchvision==0.3.0 torch==1.1.0

# Install PyTorch Detection
RUN if [ "$tag2" = "cpu" ]; then FORCE_CUDA="0"; else FORCE_CUDA="1"; fi
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python3 setup.py build develop
ENV PYTHONPATH /opt/maskrcnn-benchmark:$PYTHONPATH

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
