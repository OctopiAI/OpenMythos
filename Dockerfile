FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip build-essential curl \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install service deps
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Install torch first — flash-attn's setup.py imports torch at build time
# so torch must exist before pip tries to collect flash-attn metadata
RUN pip install --no-cache-dir torch==2.11.0

# Install the library + flash-attn (requires CUDA devel headers — hence the devel base image)
# Set FLASH_ATTN=0 to skip flash-attn (e.g. local CPU builds with no CUDA)
ARG FLASH_ATTN=1
COPY pyproject.toml .
COPY README.md .
COPY open_mythos/ open_mythos/
RUN if [ "$FLASH_ATTN" = "1" ]; then \
        pip install --no-cache-dir -e ".[flash]"; \
    else \
        pip install --no-cache-dir -e "."; \
    fi

# HuggingFace cache — persisted via a named volume so the tokenizer
# is only downloaded on first boot, not on every container restart.
ENV HF_HOME=/app/.cache/huggingface

COPY serve.py .

RUN mkdir -p /app/checkpoints

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
