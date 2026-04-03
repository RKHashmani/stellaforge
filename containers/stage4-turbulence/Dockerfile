# If gpu
# FROM stellaforge/base-gpu:latest

# If cpu
# FROM stellaforge/base-cpu:latest
# both seems not ready yet

# Otherwise, from scratch
FROM python:3.11-slim

# System dependencies (consolidated into one layer)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Core scientific packages
RUN pip install --no-cache-dir numpy scipy cvxpy ilupp pyamg mosek

# JAX — swap comment for CPU/GPU
# RUN pip install --no-cache-dir "jax[cpu]>=0.8,<0.9"
RUN pip install --no-cache-dir "jax[cuda12]>=0.8,<0.9"

# Turbulence packages
WORKDIR /packages
RUN git clone https://github.com/uwplasma/SPECTRAX-GK.git
WORKDIR /packages/SPECTRAX-GK

RUN pip install -e .

WORKDIR /app
ENTRYPOINT ["python", "/app/scripts/run.py"]