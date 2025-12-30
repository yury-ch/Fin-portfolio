FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-microservices.txt .
RUN pip install --no-cache-dir -r requirements-microservices.txt

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8001 8002 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
