FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DEFAULT_TIMEOUT=300 \
    POETRY_VERSION=1.8.2 \
    POETRY_HTTP_TIMEOUT=600

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry with timeout config
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && ln -s /root/.local/bin/poetry /usr/local/bin/poetry \
 && poetry config virtualenvs.create false \
 && poetry config installer.max-workers 1

WORKDIR /app

# Install torch first via pip (more reliable for large packages)
RUN pip install --timeout 600 torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install remaining dependencies (torch already installed)
RUN poetry install --no-root --no-interaction --no-ansi --only main || \
    (sleep 10 && poetry install --no-root --no-interaction --no-ansi --only main)

# Install scispaCy and the BC5CDR NER model via pip (separate from poetry to avoid dependency conflicts)
RUN pip install --no-deps scispacy==0.5.4 \
 && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Copy application code
COPY . .

# Generate gRPC stubs
RUN python generate_protos.py --proto-dir proto --out-dir generated

# Create non-root user
RUN useradd -m -u 1000 appuser \
 && mkdir -p /home/appuser/.cache /home/appuser/.stanza_resources \
 && chown -R appuser:appuser /app /home/appuser

USER appuser

# Environment
ENV PYTHONPATH=/app \
    LOCAL_MODE=true \
    PRELOAD_MODELS=true \
    STANZA_RESOURCES_DIR=/home/appuser/.stanza_resources

# Download Stanza English model
RUN python -c "import stanza; stanza.download('en')"

EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; from generated.nlp import nlp_pb2, nlp_pb2_grpc; ch=grpc.insecure_channel('localhost:50051'); stub=nlp_pb2_grpc.NLPServiceStub(ch); stub.Health(nlp_pb2.HealthRequest())"

CMD ["python", "main.py"]
