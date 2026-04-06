FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python properly
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Optional but recommended (faster installs)
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Preload model (now python exists)
RUN python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct'); \
AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8234"]