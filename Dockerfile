FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Env-configured; pass at runtime, never bake secrets into the image.
EXPOSE 8080
USER nobody
# Liveness probe against /healthz. Assumes the default PORT=8080 and plain WS
# (no native TLS); override or disable when you change either.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["deepgram-msteams-bridge"]
