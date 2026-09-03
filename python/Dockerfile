FROM python:3.13.3-slim-bookworm AS python-wheel-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY python/docker-build-requirements.txt /tmp/docker-build-requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/docker-build-requirements.txt

WORKDIR /src
COPY thirdparty ./thirdparty
COPY src ./src
COPY python ./python

FROM python-wheel-base AS python-wheel-build
WORKDIR /src/python
RUN python -m build --wheel --no-isolation --outdir /out

FROM python-wheel-base AS python-wheel-test-base
ENV PYTHONPATH=/src/python
COPY --from=python-wheel-build /out /out
RUN python -m pip install --no-deps /out/simjit-*.whl

FROM python-wheel-test-base AS python-wheel-unit-test
WORKDIR /src
RUN python -m pytest -q python/tests/test.py

FROM python-wheel-test-base AS python-wheel-e2e-test
WORKDIR /src
RUN python -m pytest -q python/tests/e2e_test.py

FROM scratch AS export-wheel
COPY --from=python-wheel-build /out/ /
