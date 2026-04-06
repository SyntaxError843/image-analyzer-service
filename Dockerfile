FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    poppler-utils \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- app deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache \
              /usr/local/lib/python3*/dist-packages/**/tests \
              /usr/local/lib/python3*/dist-packages/**/__pycache__

COPY app ./app

ENV NVIDIA_VISIBLE_DEVICES=all
EXPOSE 8234

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8234"]
