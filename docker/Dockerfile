# Deep-Spell-9 v2 service image (CPU inference).
# Build multi-arch:
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t ds9:2.0 .
# Run (models/config mounted):
#   docker run --rm -p 8091:8091 -v $PWD/artifacts:/data ds9:2.0 ds9 serve /data/service.json

FROM python:3.12-slim AS base

WORKDIR /ds9
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

EXPOSE 8091
CMD ["ds9", "--help"]
