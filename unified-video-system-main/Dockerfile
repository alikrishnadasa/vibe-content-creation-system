# RunPod Serverless Docker Image for Unified Video System
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install RunPod SDK
RUN pip3 install runpod

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/output /app/temp /app/cache

# Set permissions
RUN chmod +x /app/main.py

# Create RunPod handler
COPY runpod_handler.py /app/

# Expose port (optional, mainly for local testing)
EXPOSE 8080

# Start command for RunPod serverless
CMD ["python3", "-u", "runpod_handler.py"]