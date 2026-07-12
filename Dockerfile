FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Env-configured; pass at runtime, never bake secrets into the image.
EXPOSE 8080
USER nobody
CMD ["deepgram-msteams-bridge"]
