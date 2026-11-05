# inferop deployment image: build + run all standalone tests.
# GPU sections auto-skip when no device is visible, so this image also builds
# (and CPU-tests) on machines without an NVIDIA GPU.
#
#   docker build -t inferop -f docker/Dockerfile .
#   docker run --rm --gpus all inferop            # full tests on a GPU host
#   docker run --rm inferop                        # CPU golden tests only
#
ARG BASE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE}

ARG INFEROP_CUDA_ARCHES="90"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake ninja-build ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/inferop
COPY . .

RUN cmake -B build -G Ninja -DINFEROP_CUDA_ARCHES="${INFEROP_CUDA_ARCHES}" \
  && cmake --build build -j

# benchmark binary is included; override CMD to run it instead of tests
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
