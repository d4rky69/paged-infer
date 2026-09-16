FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast package management
RUN pip install --no-cache-dir uv

# Copy project definition
COPY pyproject.toml README.md ./
COPY paged_infer ./paged_infer

# Install dependencies and project in editable mode
RUN uv pip install --system -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "paged_infer.api.server", "--host", "0.0.0.0", "--port", "8000"]
