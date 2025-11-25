# ===========================
# 1) Base image
# ===========================
FROM python:3.12

WORKDIR /app

# ===========================
# 2) System dependencies
# ===========================
RUN apt-get update && apt-get install -y \
	ffmpeg \
	libsndfile1 \
	git \
	&& apt-get clean

# ===========================
# 3) Copy project
# ===========================
COPY . /app

# ===========================
# 4) Install Python deps
# ===========================
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r models/agent/requirements.txt
RUN pip install --no-cache-dir -r api/model_apirequirements.txt
RUN pip install --no-cache-dir -r api/llm_api/requirements.txt || true

# ===========================
# 5) Pre-build RAG index (optional)
# ===========================
RUN python models/agent/build_index.py || true

# ===========================
# 6) Default app (Web UI)
# ===========================
EXPOSE 8000

CMD ["python", "models/agent/web_app.py"]
