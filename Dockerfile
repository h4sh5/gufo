# syntax=docker/dockerfile:1
#
# Gufo for AMD Strix Halo: compile the gfx1151 engine, then package it for
# Podman.
#
#   podman build -t gufo:local .
#   podman run --rm \
#     --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     --ulimit memlock=-1 \
#     -p 8080:8080 -v ./models:/models:ro \
#     gufo:local serve llm --model /models/model.gguf \
#       --speculative dflash2 --dflash-model /models/draft.gguf
#
# The image carries the ROCm 10.1 userspace that the flake pins (HIP Clang 24
# and hipBLAS/hipBLASLt/rocBLAS with gfx1151 kernels). The amdgpu kernel driver,
# its parameters and the memory settings stay on the host: pass /dev/kfd and
# /dev/dri, keep --ulimit memlock=-1 so the prompt cache can pin memory, and on
# SELinux hosts allow container device mapping. Rootless Podman needs crun for
# --group-add keep-groups; see README.md#quickstart.
#
# The Nix build patches ROCr so queue save/restore buffers and HIP staging stay
# resident during host memory compaction (.devops/nix/patches). AMD's image
# ships an unpatched ROCr, which is the ordinary configuration for a container
# with unlimited memlock; apply the patch when rebuilding the runtime itself.

ARG ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:10.1.0-full

# ---------------------------------------------------------------------------
# Build: the release preset, i.e. RelWithDebInfo with ENGINE_ENABLE_HIP=ON and
# CMAKE_HIP_ARCHITECTURES=gfx1151, as `nix build` produces.
# ---------------------------------------------------------------------------
FROM ${ROCM_IMAGE} AS build

# AMD's ROCm image has no ld.so entry for /opt/rocm/lib, so link-time and
# post-build checks need the path explicitly.
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/opt/rocm/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config ca-certificates \
      libicu-dev libcurl4-openssl-dev libssl-dev \
      libpng-dev libjpeg-dev libwebp-dev \
      libsndfile1-dev libsoxr-dev \
      libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# AMD's ROCm dev image already installs the HIP toolchain and the matrix
# libraries Gufo needs. Fail here with a readable message rather than in the
# middle of CMake configuration. On a plain base image, install the ROCm
# development packages from AMD's repository first (hipblas-dev,
# hipblaslt-dev, rocblas-dev, hipcub-dev, rocprim-dev, rocwmma-dev on the
# classic rocm apt repo; ROCm 10 ships them through amdrocm-* packages).
RUN set -eu; \
    for cfg in hip hipblas hipblaslt rocblas; do \
      test -d "/opt/rocm/lib/cmake/$cfg" || { echo "missing $cfg CMake config"; exit 1; }; \
    done; \
    for h in hipcub/hipcub.hpp rocprim/rocprim.hpp rocwmma/rocwmma.hpp; do \
      test -f "/opt/rocm/include/$h" || { echo "missing $h"; exit 1; }; \
    done; \
    /opt/rocm/llvm/bin/clang++ --version | head -1; \
    /opt/rocm/bin/hipcc --version | head -1

WORKDIR /src

# One COPY: HIP kernels are compiled from the whole tree, so splitting the
# sources rarely saves a rebuild.
COPY . .

ARG GUFO_REVISION=container
# 0 means "use every core"; pass --build-arg BUILD_JOBS=4 to stay responsive.
ARG BUILD_JOBS=0

RUN set -eux; \
    jobs="${BUILD_JOBS}"; \
    if ! [ "${jobs}" -gt 0 ] 2>/dev/null; then jobs="$(nproc)"; fi; \
    cmake --preset release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DGUFO_RELEASE_VERSION="$(cat version.txt)" \
      -DGUFO_REVISION="${GUFO_REVISION}"; \
    cmake --build --preset release --parallel "${jobs}"; \
    # The build ran without a GPU: hipcc compiles gfx1151 code offline, so only
    # serving needs /dev/kfd.
    ./build/release/gufo --version; \
    DESTDIR=/stage cmake --install build/release; \
    ! ldd /stage/usr/local/bin/gufo | grep 'not found'

# ---------------------------------------------------------------------------
# Runtime: ROCm userspace, ffmpeg for generated media, the installed engine.
# ---------------------------------------------------------------------------
FROM ${ROCM_IMAGE} AS runtime

ARG GUFO_RELEASE_VERSION=development
ARG GUFO_REVISION=container

LABEL org.opencontainers.image.title="gufo" \
      org.opencontainers.image.description="Gufo inference engine for AMD Strix Halo (gfx1151)" \
      org.opencontainers.image.source="https://github.com/franzmoca/gufo" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${GUFO_RELEASE_VERSION}" \
      org.opencontainers.image.revision="${GUFO_REVISION}" \
      io.gufo.gpu="gfx1151" \
      io.gufo.rocm="10.1"

ENV ROCM_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    # Disk prompt cache and downloaded weights land under HOME/.cache.
    HOME=/var/lib/gufo \
    # Read by `gufo serve` when --host/--port are absent, and by the healthcheck.
    GUFO_HOST=0.0.0.0 \
    GUFO_PORT=8080

# Runtime libraries for tokenization, HTTPS, hashing, image/audio/video
# decoding, plus the ffmpeg and ffprobe executables Gufo invokes.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      libicu74 libcurl4t64 libssl3t64 \
      libpng16-16t64 libjpeg-turbo8 libwebp7 \
      libsndfile1 libsoxr0 \
      libavformat60 libavcodec60 libavutil58 libswscale7 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/rocm/lib > /etc/ld.so.conf.d/99-rocm.conf && ldconfig \
    # Serve as uid/gid 1000, which the base image already reserves for its
    # unprivileged account and the README maps with
    # --userns=keep-id:uid=1000,gid=1000. Writable cache lands in HOME.
    && mkdir -p /models /var/lib/gufo/.cache \
    && chown -R 1000:1000 /var/lib/gufo

COPY --from=build /stage/usr/local/ /usr/local/

# Guard against a missing runtime library at serve time.
RUN set -eux; \
    ldconfig; \
    ! ldd /usr/local/bin/gufo | grep 'not found'; \
    gufo --version; \
    gufo --help >/dev/null; \
    gufo-server --version; \
    ffmpeg -version >/dev/null; \
    ffprobe -version >/dev/null

WORKDIR /var/lib/gufo
# Never serve as root; rootless Podman maps this uid through --userns=keep-id.
USER 1000:1000

EXPOSE 8080

# Liveness only: /ready stays 503 until the model finishes loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${GUFO_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/gufo"]
CMD ["serve", "llm", "--host", "0.0.0.0", "--port", "8080"]
