FROM python:3.10.11-slim-bullseye

ARG REQUIREMENTS_FILE=requirements-docker.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-frozen-venv310.txt requirements-docker.txt requirements-venv310-core.txt requirements-venv310.txt ./

RUN python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade pip setuptools wheel \
    && "${VIRTUAL_ENV}/bin/pip" install -r "${REQUIREMENTS_FILE}"

COPY . .

RUN mkdir -p /app/data /app/models /app/config /app/data/cache/tts /app/data/cache/web

CMD ["/app/.venv/bin/python", "services/orchestrator/alon_orchestrator.py"]
