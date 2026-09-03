ARG PYTHON_IMAGE=python:3.13-slim-bookworm
ARG LLVM_MAJOR=22
ARG LLVM_DEBIAN_CODENAME=bookworm

FROM ${PYTHON_IMAGE} AS native-apt-deps

ARG LLVM_MAJOR=22
ARG LLVM_DEBIAN_CODENAME=bookworm
ARG LLVM_CONFIG=llvm-config-22
ARG LLVM_CLANG=clang-22
ARG EXTRA_APT_PACKAGES=""

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV LLVM_CONFIG=${LLVM_CONFIG}
ENV LLVM_CLANG=${LLVM_CLANG}
ENV CC=${LLVM_CLANG}
ENV CXX=clang++-${LLVM_MAJOR}

RUN install -d -m 0755 /etc/apt/keyrings && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    wget \
    && wget -qO /etc/apt/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key && \
    echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.asc] http://apt.llvm.org/${LLVM_DEBIAN_CODENAME}/ llvm-toolchain-${LLVM_DEBIAN_CODENAME}-${LLVM_MAJOR} main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang-${LLVM_MAJOR} \
    cmake \
    llvm-${LLVM_MAJOR}-dev \
    ninja-build \
    rsync \
    ${EXTRA_APT_PACKAGES} \
    && ln -sf /usr/bin/clang-${LLVM_MAJOR} /usr/local/bin/clang && \
    ln -sf /usr/bin/llvm-config-${LLVM_MAJOR} /usr/local/bin/llvm-config && \
    rm -rf /var/lib/apt/lists/*


FROM native-apt-deps AS native-python-deps

WORKDIR /app
COPY python/docker-build-requirements.txt /tmp/python-build-requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/python-build-requirements.txt


FROM native-python-deps AS native-build

ARG BUILD_WORKERS=2

COPY Makefile CMakeLists.txt CMakePresets.json ./
COPY cmake ./cmake
COPY explorer/CMakeLists.txt ./explorer/CMakeLists.txt
COPY explorer/probe ./explorer/probe
COPY python ./python
COPY scripts ./scripts
COPY src ./src
COPY thirdparty ./thirdparty

RUN NO_LTO=1 PY_LLVM=1 PY_CPP=1 WORKERS=${BUILD_WORKERS} make py && \
    NO_LTO=1 WORKERS=${BUILD_WORKERS} scripts/dev explorer-probe

COPY explorer ./explorer

RUN PYTHONPATH=/app/build/python-dev:/app/python:/app/python/src:/app \
    python -c "import explorer_llvm_probe, simjit; print('simjit native build ok')"


FROM ${PYTHON_IMAGE} AS runtime-apt-deps

ARG LLVM_MAJOR=22
ARG LLVM_DEBIAN_CODENAME=bookworm
ARG LLVM_CONFIG=llvm-config-22
ARG LLVM_CLANG=clang-22
ARG EXTRA_APT_PACKAGES=""

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/build/python-dev:/app
ENV SIMJIT_EXPLORER_TARGETS_FILE=/etc/simjit/explorer.yaml
ENV SIMJIT_EXPLORER_HOST=127.0.0.1
ENV SIMJIT_EXPLORER_PORT=8001
ENV SIMJIT_EXPLORER_ROOT_PATH=/explorer
ENV LLVM_CONFIG=${LLVM_CONFIG}
ENV LLVM_CLANG=${LLVM_CLANG}

RUN install -d -m 0755 /etc/apt/keyrings && \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    gnupg \
    wget \
    && wget -qO /etc/apt/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key && \
    echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.asc] http://apt.llvm.org/${LLVM_DEBIAN_CODENAME}/ llvm-toolchain-${LLVM_DEBIAN_CODENAME}-${LLVM_MAJOR} main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    clang-${LLVM_MAJOR} \
    g++ \
    libbenchmark-dev \
    llvm-${LLVM_MAJOR}-dev \
    nginx \
    ${EXTRA_APT_PACKAGES} \
    && ln -sf /usr/bin/clang-${LLVM_MAJOR} /usr/local/bin/clang && \
    ln -sf /usr/bin/llvm-config-${LLVM_MAJOR} /usr/local/bin/llvm-config && \
    rm -rf /var/lib/apt/lists/*


FROM runtime-apt-deps AS runtime-python-deps

WORKDIR /app
COPY explorer/requirements.txt /tmp/explorer-requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/explorer-requirements.txt


FROM runtime-python-deps AS runtime

COPY --from=native-build /app/build/python-dev ./build/python-dev
COPY Makefile CMakeLists.txt CMakePresets.json ./
COPY cmake ./cmake
COPY python ./python
COPY scripts ./scripts
COPY src ./src
COPY thirdparty ./thirdparty
COPY explorer ./explorer
COPY explorer/docker/robots.txt /usr/share/nginx/html/robots.txt
COPY explorer/app/static/simjit-favicon.svg /usr/share/nginx/html/favicon.svg
COPY explorer/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY explorer/docker/explorer.yaml /etc/simjit/explorer.yaml
COPY explorer/docker/entrypoint.sh /usr/local/bin/simjit-explorer-entrypoint
RUN chmod +x /usr/local/bin/simjit-explorer-entrypoint && \
    rm -f /etc/nginx/sites-enabled/default && \
    nginx -t

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1/', timeout=3).read(1)"

CMD ["/usr/local/bin/simjit-explorer-entrypoint"]
