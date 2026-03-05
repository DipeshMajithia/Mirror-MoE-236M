FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy requirements first for caching
COPY --chown=user requirements.txt .

# Install Python dependencies (Gradio LAST to ensure our version)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir safetensors tokenizers numpy "huggingface_hub<0.25.0" && \
    pip install --no-cache-dir gradio==4.44.1

# Copy application code
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
