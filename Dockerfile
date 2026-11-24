FROM mambaorg/micromamba:1.5.1
ARG IMAGE_VERSION=dev
ARG PACKAGE_VERSION=0.0.0
LABEL org.opencontainers.image.source="https://github.com/LBNL-UCB-STI/IMPACTS"
LABEL org.opencontainers.image.title="impacts"
LABEL org.opencontainers.image.description="IMPACTS workflow container"
LABEL org.opencontainers.image.version="$IMAGE_VERSION"

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER src /work/src
COPY README.md /work/README.md
COPY LICENSE.txt /work/LICENSE.txt
COPY setup.cfg /work/setup.cfg
COPY pyproject.toml /work/pyproject.toml
WORKDIR /work
RUN SETUPTOOLS_SCM_PRETEND_VERSION="$PACKAGE_VERSION" micromamba run -n base python -m pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/opt/conda/lib
WORKDIR /app
RUN mkdir -p /app/workspace /app/beam_input /app/impacts_cache /app/inmap
