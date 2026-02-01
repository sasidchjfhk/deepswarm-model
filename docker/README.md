# Docker Deployment

## Production Docker Setup

### 1. Training Container

```dockerfile
# docker/Dockerfile.train
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Run training
CMD ["python", "scripts/train.py"]
```

### 2. Serving Container

```dockerfile
# docker/Dockerfile.serve
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Minimal dependencies for serving
RUN pip install --no-cache-dir \
    onnxruntime \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    loguru

# Copy deployment code and model
COPY src/deployment/ src/deployment/
COPY models/onnx/ models/onnx/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.deployment.serving:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.serve
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/onnx/model.onnx
      - MODEL_VERSION=1.0.0
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

## Usage

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f model-server

# Stop services
docker-compose down

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[...]]}'
```

## Production Considerations

1. **Resource Limits**: Add CPU/memory limits in docker-compose.yml
2. **Health Checks**: Configure appropriate intervals
3. **Logging**: Use centralized logging (ELK stack)
4. **Secrets**: Use Docker secrets for sensitive data
5. **Networking**: Use custom networks for isolation
6. **Scaling**: Use Docker Swarm or Kubernetes for horizontal scaling
