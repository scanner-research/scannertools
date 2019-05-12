#!/bin/bash
set -e

yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"

if [ "${TAG}" = "cpu" ];
then
    FORCE_CUDA = 0;
else
    FORCE_CUDA = 1;
fi

docker build \
       -t ${DOCKER_REPO}:${TAG}-latest \
       --build-arg tag=${TAG} \
       --build-arg tag2=${TAG} \
       --build-arg force_cuda=${FORCE_CUDA} \
       .

if [ "${TAG}" = "cpu" ];
then
   docker run ${DOCKER_REPO}:${TAG}-latest bash \
          -c "adduser --disabled-password --gecos \"\" user && chown -R user /opt/scannertools && su -c \"cd /opt/scannertools && ./scripts/test-all.sh\" user"
fi


if [[ "$TRAVIS_BRANCH" = "master" ]]; then
    TRAVIS_TAG="latest"
fi

docker push ${DOCKER_REPO}:${TAG}-${TRAVIS_TAG}
