FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Preload model
RUN python -c "from transformers import AutoProcessor; \
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]