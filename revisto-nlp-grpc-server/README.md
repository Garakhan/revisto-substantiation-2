# NLP gRPC Service

Async gRPC service providing NLP operations: text embedding, named entity recognition, numeric entity extraction, and sentence segmentation.

## Services

| RPC | Description | Model |
|-----|-------------|-------|
| `Encode` | Text to vector embeddings | SentenceTransformer (configurable) |
| `ExtractEntities` | Named entity recognition | spaCy or HuggingFace (configurable) |
| `ExtractNumericEntities` | Numeric entity extraction (CARDINAL, PERCENT, MONEY, etc.) | Stanza NER |
| `Sentencise` | Sentence segmentation | Stanza tokenizer |
| `Health` | Model health check with dummy inference | - |

Each service can be independently enabled/disabled via settings.

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Poetry

### Installation

```bash
poetry install
```

### Generate gRPC code

```bash
make generate-all
```

### Run the server

```bash
python main.py
```

The server starts on `localhost:50051` by default. All enabled models are preloaded concurrently at startup.

## Usage

```python
import asyncio
import grpc
from generated.nlp import nlp_pb2, nlp_pb2_grpc


SERVER = "localhost:50051"
TEXTS = [
    "Barack Obama was the 45th president of the United States. He was a guy born in Hawaii???.",
    "Aspirin 500mg tablets cost $12.99 for a pack of 30. Buy it or not to buy it, that is the question.",
]


async def main():
    async with grpc.aio.insecure_channel(SERVER) as channel:
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        # # 1. Health check
        print("=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)
        health = await stub.Health(nlp_pb2.HealthRequest())
        print(f"Healthy: {health.healthy}")
        print(f"  Embedding:   enabled={health.embedding.enabled}, ready={health.embedding.ready}")
        print(f"  NER:         enabled={health.ner.enabled}, ready={health.ner.ready}")
        print(f"  Sentenciser: enabled={health.sentenciser.enabled}, ready={health.sentenciser.ready}")
        print(f"  Numeric NER: enabled={health.numeric_ner.enabled}, ready={health.numeric_ner.ready}")

        # 2. Encode
        print(f"\n{'=' * 60}")
        print("ENCODE")
        print("=" * 60)
        try:
            resp = await stub.Encode(nlp_pb2.EmbedRequest(texts=TEXTS))
            print(f"Dimension: {resp.dimension}")
            for i, emb in enumerate(resp.embeddings):
                print(f"  Text {i}: [{emb.values[0]:.4f}, {emb.values[1]:.4f}, ...] (len={len(emb.values)})")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 3. Extract Entities (NER)
        print(f"\n{'=' * 60}")
        print("EXTRACT ENTITIES")
        print("=" * 60)
        try:
            resp = await stub.ExtractEntities(nlp_pb2.NERRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}: {TEXTS[i][:50]}...")
                for ent in result.entities:
                    print(f"    - {ent.text!r} [{ent.type}] score={ent.score:.2f} ({ent.start}:{ent.end})")
                if not result.entities:
                    print("    (no entities)")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 4. Extract Numeric Entities
        print(f"\n{'=' * 60}")
        print("EXTRACT NUMERIC ENTITIES")
        print("=" * 60)
        try:
            resp = await stub.ExtractNumericEntities(nlp_pb2.NumericRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}: {TEXTS[i][:50]}...")
                for ent in result.entities:
                    print(f"    - {ent.text!r} [{ent.type}] ({ent.start}:{ent.end})")
                print(f"    Tokens: {list(result.tokens)}")
                if not result.entities:
                    print("    (no numeric entities)")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")

        # 5. Sentencise
        print(f"\n{'=' * 60}")
        print("SENTENCISE")
        print("=" * 60)
        try:
            resp = await stub.Sentencise(nlp_pb2.SentenciseRequest(texts=TEXTS))
            for i, result in enumerate(resp.results):
                print(f"  Text {i}:")
                for j, sent in enumerate(result.sentences):
                    print(f"    Sentence {j}: {sent!r}")
        except grpc.aio.AioRpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")


if __name__ == "__main__":
    asyncio.run(main())

```

## Configuration

All settings are in `settings.py` and can be overridden via environment variables or AWS Secrets Manager.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | SentenceTransformer model name |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding model (`cpu` or `cuda`) |
| `EMBEDDING_ENABLED` | `True` | Enable/disable embedding service |
| `NER_MODEL` | `d4data/biomedical-ner-all` | NER model (spaCy name or HuggingFace path) |
| `NER_ENABLED` | `True` | Enable/disable NER service |
| `STANZA_LANG` | `en` | Stanza language for sentenciser and numeric NER |
| `SENTENCISER_ENABLED` | `True` | Enable/disable sentenciser |
| `NUMERIC_NER_ENABLED` | `True` | Enable/disable numeric NER |
| `LOCAL_MODE` | `True` | Use insecure port (set `False` for TLS) |

## Testing

Tests automatically connect to a running server if available and healthy. Otherwise, they fall back to an in-process server with mocked models.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=services --cov-report=term
```

## Project Structure

```
.
├── proto/                     # Protocol Buffer definitions
│   └── nlp.proto              # NLP service definition
├── generated/                 # Generated gRPC code
│   ├── nlp/                   # Standard stubs
│   └── pythonic_stubs/        # BetterProto async stubs
├── services/
│   ├── nlp_services.py        # gRPC service implementation
│   └── src/
│       ├── helpers.py          # Utility functions
│       └── constants.py        # Constants (numeric entity types)
├── tests/
│   ├── conftest.py            # Pytest fixtures (live/mock server detection)
│   └── integration/
│       └── test_nlp_services.py
├── main.py                    # Server entry point
├── settings.py                # Configuration
├── generate_protos.py         # Proto code generation script
├── Makefile                   # Build commands
└── test.py                    # Manual gRPC client script
```

## Code Generation

```bash
make generate-all           # Standard + BetterProto stubs
make protoc-generate        # Standard stubs only
make generate-pythonic-stubs  # BetterProto stubs only
```

## Docker

```bash
docker-compose up --build
```

## Links

- [gRPC Python](https://grpc.io/docs/languages/python/)
- [SentenceTransformers](https://www.sbert.net/)
- [Stanza](https://stanfordnlp.github.io/stanza/)
