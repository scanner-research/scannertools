ARG tag
FROM scannerresearch/scanner:${tag}-latest

WORKDIR /opt
COPY . scannertools
RUN cd scannertools && pip3 install --upgrade setuptools && python3 setup.py install
RUN git clone https://github.com/davidsandberg/facenet && \
    git clone https://github.com/scanner-research/rude-carnie
ENV PYTHONPATH /opt/facenet/src:/opt/rude-carnie:$PYTHONPATH

WORKDIR /app
