# Use official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your training code
COPY trainer/ ./trainer/

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Disable TorchDynamo globally via environment variables
ENV TORCHDYNAMO_DISABLE=1
ENV DISABLE_TORCH_DYNAMO=1
ENV TORCH_COMPILE_DISABLE=1
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Run your training script
ENTRYPOINT ["python3", "trainer/file_name.py"]
