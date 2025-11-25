##############################
# Stage 1 — Base Image + Deps
##############################
FROM python:3.12-slim AS base

WORKDIR /app

ENV PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR="/tmp/pipcache"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements for caching
COPY requirements.txt ./requirements.txt
COPY models/agent/requirements.txt ./agent-req.txt
COPY api/requirements.txt ./api-req.txt

# Install heavy deps ONCE (cached)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r agent-req.txt && \
    pip install --no-cache-dir -r api-req.txt


##############################
# Stage 2 — Runtime
##############################
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy installed python deps from base
COPY --from=base /usr/local /usr/local

# Copy source code
COPY . /app

# Pre-build RAG index (cached if unchanged)
RUN python models/agent/build_index.py || true

EXPOSE 8000

CMD ["python", "models/agent/web_app.py"]
