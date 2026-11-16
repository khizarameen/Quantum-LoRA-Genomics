FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system build/runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep image size smaller, then remaining dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy project code and saved fine-tuned model / Quantum-LoRA weights
COPY app.py check.py predict_promoter.py finetune_dnabert.py quantum_lora_genomics.py ./
COPY my_fine_tuned_dnabert/ ./my_fine_tuned_dnabert/
COPY quantum_lora_adapters.pt ./quantum_lora_adapters.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
