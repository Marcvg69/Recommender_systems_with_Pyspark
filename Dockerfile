FROM python:3.10-slim-bookworm

# Java for Spark + useful tools + math libs (BLAS) for faster linear algebra
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless curl ca-certificates procps \
        libopenblas0 libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# bring in the project
COPY . .
