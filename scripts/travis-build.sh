#!/bin/bash
set -e

yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"

docker build \
       -t ${DOCKER_REPO}:${TAG}-latest \
       --build-arg tag=${TAG} \
       --build-arg tag2=${TAG} \
       .

if [ "${TAG}" = "cpu" ];
then
   docker run ${DOCKER_REPO}:${TAG}-latest bash \
          -c "cd /opt/scannertools && python3 setup.py test"
fi

docker push ${DOCKER_REPO}:${TAG}-latest
