# Revisto Evidence Aligned

A modular system for indexing reference documents and searching evidence to support claims using semantic alignment.

## Features

- PDF parsing with LandingAI
- Semantic search with BioBERT embeddings
- Elasticsearch hybrid search (vector + BM25)
- Named Entity Recognition for biomedical entities
- Claude-powered evidence filtering and ranking
- REST API with FastAPI

## Project Structure

```
revisto_evidence_aligned/
├── api/            # FastAPI application
│   ├── routers/    # API endpoints
│   └── models/     # Request/response models
├── cli/            # Command-line interfaces
│   ├── index.py    # Index reference documents
│   ├── search.py   # Search for claims
│   └── search_formatted.py  # Search with Excel output
├── config/         # Configuration management
│   └── settings.py # All configuration dataclasses
├── core/           # Core business logic
│   ├── indexing.py # Document indexing pipeline
│   ├── searching.py# Claim search pipeline
│   └── scoring.py  # Scoring algorithms
├── nlp/            # NLP processing modules
│   ├── embeddings.py    # BioBERT embedding generation
│   ├── ner.py           # Named entity recognition
│   ├── segmentation.py  # Text segmentation
│   └── tokenization.py  # Text tokenization
├── storage/        # Storage backends
│   ├── elasticsearch.py # ES operations
│   └── mappings.py      # ES index mappings
└── utils/          # Utility functions
    ├── evidence_filter.py     # Claude filtering with source priority
    ├── evidence_formatting.py # Evidence output formatting
    ├── landingai_parser.py    # PDF parsing
    ├── text_utils.py          # Encoding fixes
    └── secrets.py             # AWS Secrets Manager
```

## Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- AWS credentials (for cloud deployment)

### Local Development Setup

1. **Clone and install dependencies:**

```bash
cd revisto_evidence_aligned_clean
poetry install
```

2. **Download NLP models:**

```bash
# Stanza English model (for sentence tokenization)
python -c "import stanza; stanza.download('en')"
```

3. **Create environment file:**

```bash
cp .env.example .env
# Edit .env with your settings
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ES_URL` | Elasticsearch URL | `http://localhost:9200` |
| `ES_INDEX` | Index name | `revisto_evidence` |
| `ES_ALIAS` | Index alias | `revisto_evidence` |

### Optional (with defaults)

| Variable | Description | Default |
|----------|-------------|---------|
| `ES_USER` | ES username | `elastic` |
| `ES_PASS` | ES password | `` |
| `ES_VERIFY_CERTS` | Verify SSL certs | `false` (local), `true` (cloud) |
| `EMBED_MODEL` | Embedding model | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` |
| `NER_MODEL` | NER model | `d4data/biomedical-ner-all` |
| `NER_ENABLED` | Enable NER | `true` |
| `BM25_NORM` | BM25 score normalization | `1.0` |
| `SCORE_THRESHOLD` | Minimum relevance score | `0.40` |
| `TOPK` | Max results per search | `20` |
| `CLAUDE_FILTER_ENABLED` | Enable Claude filtering | `true` |
| `CLAUDE_MODEL` | Claude model | `claude-opus-4-20250514` |

### API Keys (via AWS Secrets Manager or environment)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key (for metadata extraction) |
| `LANDINGAI_API_KEY` | LandingAI API key (for PDF parsing) |

## Deployment

### Local Deployment (with local Elasticsearch)

1. **Start services:**

```bash
docker-compose -f docker-compose.local.yml up --build
```

This starts:
- Elasticsearch on port 9200
- Kibana on port 5601 (optional, for debugging)
- API on port 8888

2. **Verify Elasticsearch is running:**

```bash
curl http://localhost:9200/_cluster/health
```

3. **Access the API:**

- API: http://localhost:8888
- Docs: http://localhost:8888/docs
- Kibana: http://localhost:5601

### Cloud Deployment (AWS OpenSearch / Elastic Cloud)

1. **Set up AWS credentials:**

```bash
# Option 1: AWS credentials file
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

2. **Store secrets in AWS Secrets Manager:**

Create secrets with these names:
- `revisto/elasticsearch` - JSON with `url`, `username`, `password`
- `revisto/anthropic` - JSON with `api_key`
- `revisto/openai` - JSON with `api_key`
- `revisto/landingai` - JSON with `api_key`

3. **Set environment variables:**

```bash
export ES_INDEX=your_index_name
export ES_ALIAS=your_alias_name
```

4. **Start the API:**

```bash
docker-compose -f docker-compose.cloud.yml up --build
```

### gRPC Deployment (Separate NLP Service)

For production environments where you want to:
- Scale NLP workloads independently from the API
- Reduce API container memory footprint
- Share NLP models across multiple API instances

1. **Start the services:**

```bash
docker-compose -f docker-compose.grpc.yml up --build
```

This starts:
- NLP gRPC service on port 50051 (hosts embedding and NER models)
- API on port 8888 (connects to NLP service via gRPC)

2. **Environment variables for gRPC mode:**

| Variable | Description | Default |
|----------|-------------|---------|
| `NLP_MODE` | NLP model mode (`local` or `grpc`) | `local` |
| `NLP_GRPC_HOST` | gRPC service hostname | `nlp-grpc` |
| `NLP_GRPC_PORT` | gRPC service port | `50051` |

3. **Access the API:**

- API: http://localhost:8888
- Docs: http://localhost:8888/api/docs
- NLP gRPC: localhost:50051

### Local + gRPC Deployment (Local ES with Separate NLP Service)

For local development with the gRPC NLP architecture:
- Local Elasticsearch for easy debugging
- Separate NLP service for testing gRPC integration
- Lighter API container (no ML models loaded)

1. **Start the services:**

```bash
docker-compose -f docker-compose.local-grpc.yml up --build
```

This starts:
- Elasticsearch on port 9200
- Kibana on port 5601 (for debugging)
- NLP gRPC service on port 50051
- API on port 8888 (connects to both ES and NLP via gRPC)

2. **Access the services:**

- API: http://localhost:8888
- Docs: http://localhost:8888/api/docs
- Kibana: http://localhost:5601
- NLP gRPC: localhost:50051

## Usage

### CLI: Index Reference Documents

```bash
# Index PDFs with sentence-level segmentation (default)
python -m revisto_evidence_aligned.cli.index \
  --refs-dir /path/to/pdfs \
  --org-id 1 \
  --brand-id 1

# Index with block-level segmentation (larger chunks)
python -m revisto_evidence_aligned.cli.index \
  --refs-dir /path/to/pdfs \
  --org-id 1 \
  --brand-id 1 \
  --segmentation-level block

# Disable NER extraction
python -m revisto_evidence_aligned.cli.index \
  --refs-dir /path/to/pdfs \
  --org-id 1 \
  --brand-id 1 \
  --no-ner
```

### CLI: Search Claims

```bash
# Search claims from JSON file
python -m revisto_evidence_aligned.cli.search \
  --claims-file claims.json \
  --org-id 1 \
  --brand-id 1 \
  --visual_claims results.json

# Search with custom threshold
python -m revisto_evidence_aligned.cli.search \
  --claims-file claims.json \
  --org-id 1 \
  --brand-id 1 \
  --threshold 0.5 \
  --topk 10
```

### CLI: Search with Excel Output

```bash
# Search and visual_claims to Excel
python -m revisto_evidence_aligned.cli.search_formatted \
  --claims-file claims.csv \
  --org-id 1 \
  --brand-id 1 \
  --visual_claims merck_results.xlsx
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/index` | Index PDF files |
| POST | `/api/v1/search` | Search for evidence |

## Claims File Format

### JSON Format

```json
{
  "claims": [
    {
      "claim_id": "1",
      "text": "The treatment showed 50% improvement in patients."
    },
    {
      "claim_id": "2",
      "text": "Side effects were minimal in clinical trials."
    }
  ]
}
```

### CSV Format

```csv
claim_id,text
1,"The treatment showed 50% improvement in patients."
2,"Side effects were minimal in clinical trials."
```

## Development

### Run tests

```bash
poetry run pytest
```

### Format code

```bash
poetry run black revisto_evidence_aligned/
poetry run ruff check revisto_evidence_aligned/
```

### Regenerate lock file

```bash
poetry lock
```

## Troubleshooting

### Elasticsearch connection issues

```bash
# Check if ES is running
curl -X GET "localhost:9200/_cluster/health?pretty"

# Check index exists
curl -X GET "localhost:9200/_cat/indices?v"
```

### Model download issues

Models are downloaded on first use. Ensure you have internet access and sufficient disk space (~2GB for all models).

```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')"
```

### Memory issues

For large documents, increase Docker memory limits in `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
```
