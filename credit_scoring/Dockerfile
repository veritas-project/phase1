FROM continuumio/miniconda3:4.8.2

ARG WORKSPACE=/workspace
WORKDIR $WORKSPACE

COPY environment.yaml .

RUN conda env update -n base -f environment.yaml && \
    conda clean -ya
