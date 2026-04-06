FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps, pip install, then remove build deps to keep image small (T-01, T-06)
COPY requirements-lock.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements-lock.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8001 8002 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
