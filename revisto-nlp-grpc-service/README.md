# Revisto NLP gRPC Service

A gRPC service for NLP operations including text embedding and Named Entity Recognition (NER).

## Features

- **Text Embedding**: Encode text to dense vector embeddings using sentence-transformers
- **Named Entity Recognition**: Extract biomedical entities using transformer models
- **Health Check**: Monitor service and model status

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Generate gRPC Stubs

```bash
python generate_protos.py --proto-dir proto --out-dir generated
```

### 3. Run the Server

```bash
# Local development (insecure)
python main.py --host 0.0.0.0 --port 50051

# Or with environment variables
export LOCAL_MODE=true
export EMBEDDING_MODEL=pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
export NER_MODEL=d4data/biomedical-ner-all
python main.py
```

## Configuration

Configuration is done via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_DEFAULT_HOST` | `0.0.0.0` | Host to bind to |
| `SERVICE_DEFAULT_PORT` | `50051` | Port to bind to |
| `LOCAL_MODE` | `true` | Use insecure connection (no SSL) |
| `GRPC_MAX_WORKERS` | `10` | Max thread pool workers |
| `EMBEDDING_MODEL` | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | Sentence transformer model |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding model (`cpu` or `cuda`) |
| `EMBEDDING_NORMALIZE` | `true` | Normalize embeddings |
| `NER_MODEL` | `d4data/biomedical-ner-all` | NER model name |
| `NER_ENABLED` | `true` | Enable NER extraction |
| `PRELOAD_MODELS` | `true` | Preload models on startup |

## API

### Service: `NLPService`

#### `Encode(EmbedRequest) -> EmbedResponse`

Encode texts to embeddings.

**Request:**
```protobuf
message EmbedRequest {
  repeated string texts = 1;
  bool normalize = 2;
}
```

**Response:**
```protobuf
message EmbedResponse {
  repeated FloatArray embeddings = 1;
  int32 dimension = 2;
}
```

#### `ExtractEntities(NERRequest) -> NERResponse`

Extract named entities from texts.

**Request:**
```protobuf
message NERRequest {
  repeated string texts = 1;
}
```

**Response:**
```protobuf
message NERResponse {
  repeated TextEntities results = 1;
}

message TextEntities {
  repeated Entity entities = 1;
}

message Entity {
  string text = 1;
  string type = 2;
  float score = 3;
  int32 start = 4;
  int32 end = 5;
}
```

#### `Health(HealthRequest) -> HealthResponse`

Health check endpoint.

**Response:**
```protobuf
message HealthResponse {
  bool healthy = 1;
  string embedding_model = 2;
  string ner_model = 3;
  bool models_loaded = 4;
}
```

## Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install poetry && poetry install --no-dev

EXPOSE 50051

CMD ["python", "main.py"]
```

## Client Usage

```python
from revisto_evidence_aligned.grpc_client import GrpcEmbeddingModel, GrpcNERExtractor

# Create clients
embedder = GrpcEmbeddingModel(grpc_address="localhost:50051")
ner = GrpcNERExtractor(grpc_address="localhost:50051")

# Encode texts
embeddings = embedder.encode(["Hello world", "Another text"])
print(embeddings.shape)  # (2, 768)

# Extract entities
entities = ner.extract("Aspirin is used to treat pain.")
print(entities)  # [{"text": "Aspirin", "type": "DRUG", ...}]

# Close connections
embedder.close()
ner.close()
```
