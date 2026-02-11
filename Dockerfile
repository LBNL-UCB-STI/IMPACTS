FROM mambaorg/micromamba:1.5.1
LABEL org.opencontainers.image.source="https://github.com/LBNL-UCB-STI/"

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

COPY src /work/src
COPY data /work/data
COPY README.md /work/README.md
COPY README.rst /work/README.rst
COPY setup.cfg /work/setup.cfg
COPY pyproject.toml /work/pyproject.toml

ENV PYTHONPATH=/work/src
WORKDIR /work

VOLUME ["/input", "/output"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "impacts"]
