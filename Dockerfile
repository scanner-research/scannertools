ARG tag
FROM scannerresearch/scanner:${tag}-latest
ARG tag2

WORKDIR /opt
COPY . scannertools
RUN if [ "$tag2" = "cpu" ]; then pip3 install tensorflow==1.5.0; else pip3 install tensorflow-gpu==1.5.0; fi
RUN cd scannertools && pip3 install --upgrade setuptools && python3 setup.py install
RUN git clone https://github.com/davidsandberg/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH

WORKDIR /app
