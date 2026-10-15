FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts \
    PORT=8000

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The trained model artifact is committed to the repo (see artifacts/),
# so the image serves predictions out of the box. To use a different
# model, mount a directory with model.joblib and metrics.json at
# /app/artifacts or set ARTIFACTS_DIR to its path.
COPY artifacts ./artifacts

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["sh", "-c", "uvicorn readmission.api:app --host 0.0.0.0 --port ${PORT}"]
