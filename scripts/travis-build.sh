#!/bin/bash
set -e

yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"

docker build \
       -t ${DOCKER_REPO}:${TAG}-latest \
       --build-arg tag=${TAG} .

docker run ${DOCKER_REPO}:${TAG}-latest bash \
       -c "cd /opt/scannertools && python3 setup.py test"

docker push ${DOCKER_REPO}:${TAG}-latest
