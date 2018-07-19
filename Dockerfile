ARG tag
FROM scannerresearch/scanner:${tag}-latest
ARG tag2

WORKDIR /opt
RUN if [ "$tag2" = "cpu" ]; then pip3 install tensorflow==1.5.0; else pip3 install tensorflow-gpu==1.5.0; fi
RUN git clone https://github.com/davidsandberg/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH
RUN apt-get update && apt-get install -y jq
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk kubectl
COPY . scannertools
RUN cd scannertools && pip3 install --upgrade setuptools && python3 setup.py install

WORKDIR /app
