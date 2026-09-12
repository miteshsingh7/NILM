# Hugging Face Spaces Dockerfile for NILM Telemetry Rack v4.2
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal build tools and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application and dataset files
COPY . .

# Hugging Face Spaces container networking standard
ENV PORT=7860
EXPOSE 7860

# Run the aiohttp server (respects $PORT from environment)
CMD ["python", "run_dashboard.py", "--host", "0.0.0.0"]
