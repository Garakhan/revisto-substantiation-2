# Revisto Evidence Aligned - Architecture Diagrams

## 1. High-Level System Architecture (gRPC Mode with Cloud Elasticsearch)

```mermaid
flowchart TB
    subgraph External["External Services"]
        LandingAI["LandingAI API<br/>(PDF Parsing)"]
        Claude["Anthropic Claude API<br/>(Filtering & Tables)"]
        AWS["AWS Secrets Manager<br/>(ES Credentials)"]
        S3["AWS S3<br/>(Zero Data Retention)"]
        ES["Elastic Cloud<br/>Elasticsearch 8.x<br/>(Vector + BM25 Search)"]
        Kibana["Elastic Cloud Kibana<br/>(Monitoring & Dev Tools)"]
    end

    subgraph Docker["Docker Network: revisto-net"]
        subgraph API_Container["Container: revisto-api (port 8888)"]
            Routers["API Routers<br/>/index, /search, /health"]
            Core["Core Layer<br/>indexing, searching, scoring"]
            gRPC_Client["gRPC Client Stubs<br/>GrpcEmbeddingModel<br/>GrpcNERExtractor<br/>GrpcSentenciser"]
            Utils["Utilities<br/>evidence_filter, parsers"]
        end

        subgraph NLP_Container["Container: revisto-nlp-grpc (port 50051)"]
            gRPC_Server["gRPC Server<br/>NLPService"]
            Embedder["BioBERT Embeddings<br/>(384 dims)"]
            NER["Biomedical NER<br/>d4data/biomedical-ner-all"]
            Stanza["Stanza Sentenciser"]
        end
    end

    Client["API Client<br/>(HTTP)"] -->|REST API| Routers
    Routers --> Core
    Core --> Utils
    Core --> gRPC_Client
    gRPC_Client -->|gRPC Protocol| gRPC_Server
    gRPC_Server --> Embedder
    gRPC_Server --> NER
    gRPC_Server --> Stanza
    Core -->|HTTPS + API Key| ES
    ES --- Kibana
    Utils -->|HTTPS| LandingAI
    Utils -->|HTTPS| Claude
    API_Container -->|HTTPS| AWS
    Utils -->|HTTPS| S3

    style External fill:#ffe6f0,stroke:#cc0066
    style Docker fill:#e6f3ff,stroke:#0066cc
    style API_Container fill:#d4edda,stroke:#28a745
    style NLP_Container fill:#fff3cd,stroke:#ffc107
    style ES fill:#cce5ff,stroke:#004085
    style Kibana fill:#e2d5f1,stroke:#6f42c1
```

## 2. Indexing Flow

```mermaid
flowchart LR
    subgraph Input
        PDF["PDF Files"]
        Meta["org_id, brand_id"]
    end

    subgraph Parsing["1. Parsing Phase"]
        LandingAI["LandingAI API"]
        Chunks["Extract Chunks<br/>& Metadata"]
    end

    subgraph Segmentation["2. Segmentation"]
        TextSeg["Text Segments"]
        TableSeg["Table Segments"]
        Stanza["Stanza<br/>Sentence Split"]
        Claude_Lin["Claude<br/>Table Linearizer"]
    end

    subgraph Features["3. Feature Extraction"]
        Embed["BioBERT<br/>Embeddings<br/>(384 dims)"]
        NER["Biomedical<br/>NER"]
        Numeric["Numeric<br/>Extraction"]
    end

    subgraph Storage["4. Storage"]
        ES["Elasticsearch<br/>Bulk Index"]
        Index["Index:<br/>org_{id}_brand_{id}"]
    end

    PDF --> LandingAI
    Meta --> LandingAI
    LandingAI --> Chunks
    Chunks --> TextSeg
    Chunks --> TableSeg
    TextSeg --> Stanza
    TableSeg --> Claude_Lin
    Stanza --> Embed
    Claude_Lin --> Embed
    Embed --> NER
    NER --> Numeric
    Numeric --> ES
    ES --> Index
```

## 3. Search Flow

```mermaid
flowchart LR
    subgraph Input
        Claim["Claim Text"]
        Filters["org_id, brand_id<br/>threshold, top_k"]
    end

    subgraph Features["1. Claim Features"]
        ClaimEmbed["BioBERT<br/>Embedding"]
        ClaimNER["Entity<br/>Extraction"]
        ClaimNum["Numeric<br/>Extraction"]
    end

    subgraph Search["2. Hybrid Search"]
        ES["Elasticsearch"]
        BM25["BM25<br/>Lexical"]
        Vector["Vector<br/>Similarity"]
        EntityMatch["Entity<br/>Matching"]
    end

    subgraph Scoring["3. Scoring & Ranking"]
        Normalize["Normalize<br/>Scores"]
        Weights["Apply Weights<br/>semantic, lexical<br/>numeric, entity"]
        Filter["Filter by<br/>Threshold"]
    end

    subgraph Output
        Results["Evidence Results<br/>with scores"]
    end

    Claim --> ClaimEmbed
    Claim --> ClaimNER
    Claim --> ClaimNum
    Filters --> ES
    ClaimEmbed --> ES
    ClaimNER --> ES
    ClaimNum --> ES
    ES --> BM25
    ES --> Vector
    ES --> EntityMatch
    BM25 --> Normalize
    Vector --> Normalize
    EntityMatch --> Normalize
    Normalize --> Weights
    Weights --> Filter
    Filter --> Results
```

## 4. Deployment Architectures

### 4a. Local Mode (docker-compose.local.yml)

```mermaid
flowchart TB
    subgraph Docker["Docker: revisto-net"]
        subgraph API["FastAPI + NLP Models<br/>(port 8888)"]
            FastAPI["FastAPI"]
            BioBERT["BioBERT<br/>Embeddings"]
            NER["Biomedical<br/>NER"]
            Stanza["Stanza"]
        end

        ES["Elasticsearch<br/>(port 9200)"]
        Kibana["Kibana<br/>(port 5601)"]
    end

    Client["Client"] --> FastAPI
    FastAPI --> BioBERT
    FastAPI --> NER
    FastAPI --> Stanza
    FastAPI --> ES
    ES --> Kibana

    style API fill:#ffcccc
```

### 4b. gRPC Mode - Local ES (docker-compose.local-grpc.yml)

```mermaid
flowchart TB
    subgraph Docker["Docker: revisto-net"]
        subgraph API["FastAPI (Lightweight)<br/>(port 8888)"]
            FastAPI["FastAPI"]
            gRPC_Client["gRPC Client<br/>Stubs"]
        end

        subgraph NLP["NLP gRPC Service<br/>(port 50051)"]
            gRPC_Server["gRPC Server"]
            BioBERT["BioBERT"]
            NER["NER"]
            Stanza["Stanza"]
        end

        ES["Elasticsearch<br/>(port 9200)"]
        Kibana["Kibana<br/>(port 5601)"]
    end

    Client["Client"] --> FastAPI
    FastAPI --> gRPC_Client
    gRPC_Client -->|gRPC| gRPC_Server
    gRPC_Server --> BioBERT
    gRPC_Server --> NER
    gRPC_Server --> Stanza
    FastAPI --> ES
    ES --> Kibana

    style API fill:#ccffcc
    style NLP fill:#ffffcc
```

### 4c. gRPC Mode - Cloud ES (docker-compose.grpc.yml)

```mermaid
flowchart TB
    subgraph External["External Services"]
        AWS["AWS Secrets Manager"]
        ES["Elastic Cloud<br/>Elasticsearch"]
        Kibana["Elastic Cloud<br/>Kibana"]
    end

    subgraph Docker["Docker: revisto-net"]
        subgraph API["FastAPI (Lightweight)<br/>(port 8888)"]
            FastAPI["FastAPI"]
            gRPC_Client["gRPC Client<br/>Stubs"]
        end

        subgraph NLP["NLP gRPC Service<br/>(port 50051)"]
            gRPC_Server["gRPC Server"]
            BioBERT["BioBERT"]
            NER["NER"]
            Stanza["Stanza"]
        end
    end

    Client["Client"] --> FastAPI
    FastAPI --> gRPC_Client
    gRPC_Client -->|gRPC| gRPC_Server
    gRPC_Server --> BioBERT
    gRPC_Server --> NER
    gRPC_Server --> Stanza
    FastAPI -->|HTTPS + API Key| ES
    FastAPI -->|HTTPS| AWS
    ES --- Kibana

    style External fill:#ffe6f0,stroke:#cc0066
    style API fill:#ccffcc
    style NLP fill:#ffffcc
```

### 4d. Distributed Mode - Independent NLP Service

```mermaid
flowchart TB
    subgraph External["External Services (Cloud)"]
        subgraph AWS_Services["AWS"]
            AWS["AWS Secrets Manager<br/>(ES Credentials, API Keys)"]
            S3["AWS S3<br/>(Zero Data Retention)"]
        end

        subgraph Elastic["Elastic Cloud"]
            ES["Elasticsearch 8.x<br/>(Vector + BM25 Search)"]
            Kibana["Kibana<br/>(Monitoring & Dev Tools)"]
        end

        subgraph AI_APIs["AI Services"]
            LandingAI["LandingAI API<br/>(PDF Parsing & OCR)"]
            Claude["Anthropic Claude API<br/>(Evidence Filtering<br/>& Table Linearization)"]
        end
    end

    subgraph API_Host["Host A: API Server (Lightweight)"]
        subgraph API_Container["Container: revisto-api (port 8888)"]
            Routers["API Routers<br/>/index, /search, /health"]
            Core["Core Layer<br/>indexing, searching, scoring"]
            gRPC_Client["gRPC Client Stubs<br/>GrpcEmbeddingModel<br/>GrpcNERExtractor<br/>GrpcSentenciser"]
            Utils["Utilities<br/>evidence_filter<br/>landingai_parser<br/>table_linearizer"]
        end
    end

    subgraph NLP_Host["Host B: NLP Server (GPU-enabled, Scalable)"]
        subgraph NLP_Container["Container: revisto-nlp-grpc (port 50051)"]
            gRPC_Server["gRPC Server<br/>NLPService"]
            Embedder["BioBERT Embeddings<br/>(384 dims)<br/>GPU Accelerated"]
            NER["Biomedical NER<br/>d4data/biomedical-ner-all"]
            Stanza["Stanza Sentenciser"]
        end
    end

    Client["API Client<br/>(HTTP)"] -->|REST API| Routers
    Routers --> Core
    Core --> Utils
    Core --> gRPC_Client

    gRPC_Client -->|gRPC Protocol<br/>NLP_GRPC_HOST:50051| gRPC_Server
    gRPC_Server --> Embedder
    gRPC_Server --> NER
    gRPC_Server --> Stanza

    Core -->|HTTPS + API Key| ES
    ES --- Kibana

    Utils -->|HTTPS| LandingAI
    Utils -->|HTTPS| Claude
    Utils -->|HTTPS| S3
    API_Container -->|HTTPS| AWS

    style External fill:#ffe6f0,stroke:#cc0066
    style AWS_Services fill:#fff0e6,stroke:#ff9933
    style Elastic fill:#e6f0ff,stroke:#3366cc
    style AI_APIs fill:#f0ffe6,stroke:#66cc33
    style API_Host fill:#e6f3ff,stroke:#0066cc
    style NLP_Host fill:#fff3cd,stroke:#ffc107
    style API_Container fill:#d4edda,stroke:#28a745
    style NLP_Container fill:#ffffcc,stroke:#cccc00
```

**Data Flow:**

| Flow | Source | Destination | Protocol | Purpose |
|------|--------|-------------|----------|---------|
| 1 | Client | API Server | HTTP/REST | API requests |
| 2 | API Server | NLP Server | gRPC | Embeddings, NER, Sentencization |
| 3 | API Server | Elasticsearch | HTTPS | Index & search operations |
| 4 | API Server | AWS Secrets | HTTPS | Retrieve credentials |
| 5 | API Server | LandingAI | HTTPS | PDF parsing |
| 6 | API Server | Claude API | HTTPS | Evidence filtering, table linearization |
| 7 | API Server | S3 | HTTPS | Zero data retention storage |

**Configuration for Distributed Mode:**

```bash
# ============================================
# Host A: API Server (Lightweight Container)
# ============================================
NLP_MODE=grpc
NLP_GRPC_HOST=<nlp-server-ip-or-hostname>
NLP_GRPC_PORT=50051

# Elasticsearch (credentials from AWS Secrets Manager)
ES_VERIFY_CERTS=true
ES_INDEX=evidence_index

# AWS
AWS_DEFAULT_REGION=us-east-1

# Claude filtering
CLAUDE_FILTER_ENABLED=true
CLAUDE_MODEL=claude-opus-4-20250514

# LandingAI
LANDINGAI_OUTPUT_BUCKET=claimsubstantiation

# ============================================
# Host B: NLP Server (GPU-enabled)
# ============================================
EMBEDDING_MODEL=pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
EMBEDDING_DEVICE=cuda  # GPU acceleration
NER_MODEL=d4data/biomedical-ner-all
NER_ENABLED=true
GRPC_MAX_WORKERS=10
PRELOAD_MODELS=true
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Independent Scaling** | Scale NLP service on GPU instances separately from API |
| **Cost Optimization** | Run API on cheap instances, NLP on GPU only when needed |
| **Resource Isolation** | ML models don't compete with API for memory/CPU |
| **Shared NLP Service** | Multiple API instances can share one NLP service |
| **Multi-Application** | NLP service can serve multiple different applications |
| **Easier Updates** | Update NLP models without redeploying API |

## 5. Component Dependencies

```mermaid
flowchart TD
    subgraph API_Layer["API Layer"]
        app["app.py<br/>FastAPI Factory"]
        health["routers/health.py"]
        search["routers/search.py"]
        indexing["routers/indexing.py"]
        deps["dependencies.py"]
    end

    subgraph Core_Layer["Core Layer"]
        core_index["core/indexing.py"]
        core_search["core/searching.py"]
        core_score["core/scoring.py"]
    end

    subgraph NLP_Layer["NLP Layer"]
        embeddings["nlp/embeddings.py"]
        ner["nlp/ner.py"]
        segment["nlp/segmentation.py"]
    end

    subgraph Storage_Layer["Storage Layer"]
        es_client["storage/elasticsearch.py"]
        mappings["storage/mappings.py"]
    end

    subgraph Utils_Layer["Utils Layer"]
        filter["utils/evidence_filter.py"]
        parser["utils/landingai_parser.py"]
        linearizer["utils/table_linearizer.py"]
        secrets["utils/secrets.py"]
    end

    subgraph Config_Layer["Config Layer"]
        settings["config/settings.py"]
    end

    app --> health
    app --> search
    app --> indexing
    app --> deps

    search --> core_search
    indexing --> core_index

    core_search --> core_score
    core_search --> embeddings
    core_search --> ner
    core_search --> es_client

    core_index --> embeddings
    core_index --> ner
    core_index --> segment
    core_index --> es_client
    core_index --> parser
    core_index --> linearizer

    core_score --> filter

    deps --> settings
    deps --> secrets

    es_client --> mappings
```

## 6. Data Model (Elasticsearch Document)

```mermaid
erDiagram
    EVIDENCE_DOCUMENT {
        string id PK
        string text "Main content"
        float[] vector "384-dim embedding"
        string[] numeric_tokens "Extracted numbers"
        string[] entities "Named entities"
        string[] entity_types "Entity categories"
        int org_id FK
        int brand_id FK
        int page "Page number"
        object bbox "Bounding box coords"
        string table_source "Table location if applicable"
        object metadata "Additional metadata"
    }

    INDEX {
        string name PK "org_{id}_brand_{id}"
        int shards "Default: 2"
        int replicas "Default: 1"
    }

    INDEX ||--o{ EVIDENCE_DOCUMENT : contains
```

## 7. gRPC Service Definition

```mermaid
classDiagram
    class NLPService {
        +Encode(EmbedRequest) EmbedResponse
        +ExtractEntities(NERRequest) NERResponse
        +Sentencise(SentenciseRequest) SentenceResponse
        +Health(HealthRequest) HealthResponse
    }

    class EmbedRequest {
        +string[] texts
        +bool normalize
    }

    class EmbedResponse {
        +float[][] embeddings
        +int dimension
    }

    class NERRequest {
        +string[] texts
    }

    class NERResponse {
        +Entity[][] entities
    }

    class Entity {
        +string text
        +string type
        +float score
        +int start
        +int end
    }

    NLPService --> EmbedRequest
    NLPService --> EmbedResponse
    NLPService --> NERRequest
    NLPService --> NERResponse
    NERResponse --> Entity
```

## 8. Scoring Formula

```mermaid
flowchart LR
    subgraph Inputs
        ES_Score["ES Score<br/>(BM25 + Vector)"]
        Num_Overlap["Numeric Overlap<br/>Jaccard Similarity"]
        Ent_Overlap["Entity Overlap<br/>Jaccard Similarity"]
    end

    subgraph Weights
        W_Sem["w_semantic = 3"]
        W_Lex["w_lexical = 3"]
        W_Num["w_numeric = 1"]
        W_Ent["w_entity = 1"]
    end

    subgraph Calculation
        Norm["Normalize:<br/>es_norm = es / (es + k)"]
        Combine["Weighted Sum:<br/>(es_norm × 6) +<br/>(num × 1) +<br/>(ent × 1)"]
        Final["Final Score =<br/>sum / total_weight"]
    end

    ES_Score --> Norm
    W_Sem --> Combine
    W_Lex --> Combine
    Norm --> Combine
    Num_Overlap --> Combine
    W_Num --> Combine
    Ent_Overlap --> Combine
    W_Ent --> Combine
    Combine --> Final
```

---

## How to Use These Diagrams

### Option 1: Figma
1. Install the "Mermaid to Figma" plugin from Figma Community
2. Copy any mermaid code block above
3. Use the plugin to convert to Figma shapes

### Option 2: Draw.io / Diagrams.net
1. Go to draw.io
2. Click Insert > Advanced > Mermaid
3. Paste the mermaid code

### Option 3: VS Code
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file and preview

### Option 4: Online Renderer
1. Go to https://mermaid.live/
2. Paste the mermaid code to see and export as PNG/SVG

### Option 5: GitHub/GitLab
- Both platforms render Mermaid diagrams natively in markdown files
