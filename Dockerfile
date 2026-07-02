FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ninja-build is required for flash-attn source compilation (no prebuilt wheel exists for cu13+torch2.11)
# Without ninja, torch falls back to distutils which is ~10x slower and will time out
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# numpy must exist before torch/flash-attn or torch logs a NumPy init warning
RUN pip install --no-cache-dir numpy

# torch first — flash-attn's setup.py imports torch at build time
RUN pip install --no-cache-dir torch==2.11.0

# flash-attn has no prebuilt wheel for cu13+torch2.11; source compilation takes 20-30 min
# and OOMs the T4 during Docker build. Disabled until a prebuilt wheel is available.
ARG FLASH_ATTN=0
COPY pyproject.toml .
COPY README.md .
COPY open_mythos/ open_mythos/
RUN if [ "$FLASH_ATTN" = "1" ]; then \
        pip install --no-cache-dir -e ".[flash]"; \
    else \
        pip install --no-cache-dir -e "."; \
    fi

# HuggingFace cache persisted via named volume — tokenizer downloaded once, not on every restart
ENV HF_HOME=/app/.cache/huggingface

COPY serve.py .

RUN mkdir -p /app/checkpoints

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
