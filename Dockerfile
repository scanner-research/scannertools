ARG tag
FROM scannerresearch/scanner:${tag}-latest
ARG tag2

WORKDIR /opt
# Fixes travis pip failure
RUN rm /usr/share/python-wheels/urllib3-1.13.1-py2.py3-none-any.whl && pip3 install requests[security] --upgrade
RUN if [ "$tag2" = "cpu" ]; then pip3 install tensorflow==1.5.0; else pip3 install tensorflow-gpu==1.5.0; fi
RUN pip3 install torch==0.4.1 torchvision face-alignment
RUN git clone https://github.com/davidsandberg/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH
RUN apt-get update && apt-get install -y jq
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk kubectl
# https://github.com/keras-team/keras/issues/9567#issuecomment-370887563
RUN if [ "$tag2" != "cpu" ]; then \
    apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
            libcudnn7=7.0.5.15-1+cuda9.0 \
            libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
            rm -rf /var/lib/apt/lists/*; \
    fi
COPY . scannertools
RUN cd scannertools && pip3 install --upgrade setuptools && python3 setup.py install
RUN cd scannertools/scannertools/cpp_ops && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make

WORKDIR /app
