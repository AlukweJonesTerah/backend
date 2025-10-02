# Dockerfile
FROM python:3.12-slim

# Install ffmpeg and required system libs for pydub
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Entrypoint script will write credentials and start uvicorn
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
