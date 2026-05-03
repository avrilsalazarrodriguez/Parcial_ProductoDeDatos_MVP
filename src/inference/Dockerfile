FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-dev --no-install-project
RUN uv pip install --python /app/.venv/bin/python flask gunicorn

COPY src /app/src

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "src.inference.inference:app"]