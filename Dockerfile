FROM python:3.11-slim

# Labels for HF Spaces
LABEL org.opencontainers.image.title="OpenEnv Email Triage"
LABEL org.opencontainers.image.description="Real-world email triage OpenEnv environment"
LABEL openenv.version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py      .
COPY dataset.py     .
COPY graders.py     .
COPY environment.py .
COPY server.py      .
COPY openenv.yaml   .
COPY inference.py   .
COPY static/        ./static/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Environment variables (overrideable at runtime)
ENV PORT=7860
ENV HOST=0.0.0.0

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
