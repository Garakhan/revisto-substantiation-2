# Indexing Pipeline Documentation

This document describes how reference PDFs are processed, prepared for indexing, and stored in Elasticsearch.

---

## Pipeline Overview

```
PDF File → Parse (LandingAI) → Process Chunks → Extract Features → Assemble Documents → Index (Elasticsearch)
```

**Entry Points:**
- **CLI:** `revisto-index --refs-dir ./references --org-id 1 --brand-id 1`
- **API:** `POST /api/index/references`

---

## Step 1: PDF Parsing (LandingAI)

The PDF is sent to the LandingAI API for layout-aware parsing. Results are cached as JSON files next to the source PDFs.

**Input:** PDF file path
**Output:** List of chunks with metadata

Each chunk contains:
```python
{
    "text": "The actual content extracted...",
    "chunk_type": "text" | "table" | "figure",
    "page": 0,  # 0-indexed
    "bbox": [x0, y0, x1, y1],  # coordinates
    "section": "abstract" | "body" | "methods" | "results" | ...
}
```

**Caching:** Parsed results are saved as `{pdf_name}_landingai.json` for reuse.

---

## Step 2: Parallel Chunk Processing

Chunks are processed in parallel based on their type. Processing is done using `asyncio.gather()` with a `ThreadPoolExecutor` for CPU-bound tasks.

```
                    ┌─────────────────┐
                    │  LandingAI      │
                    │  Chunks         │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Text Chunks │   │Table Chunks │   │Figure Chunks│
    │             │   │      │      │   │             │
    │ Pass-through│   │      ▼      │   │ Claude API  │
    │             │   │BeautifulSoup│   │ Clean &     │
    │             │   │ Parse HTML  │   │ Extract     │
    │             │   │      │      │   │ Claims      │
    │             │   │      ▼      │   │             │
    │             │   │ Claude API  │   │             │
    │             │   │ Generate    │   │             │
    │             │   │ Statements  │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┴─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  All Segments   │
                    └─────────────────┘
```

### 2.1 Text Chunks

Text chunks pass through without modification. They are converted to `Segment` objects:

```python
Segment(
    text="The study enrolled 500 participants...",
    page=2,
    bbox=[72.0, 150.0, 540.0, 180.0],
    section="methods",
    segment_type="text",
    metadata={}
)
```

### 2.2 Table Chunks

Table linearization is a two-step process:

1. **BeautifulSoup HTML Parsing**: Extract table structure (headers and rows) from HTML
2. **Claude LLM Generation**: Convert structured data into declarative natural language statements

**Step 1: HTML Parsing with BeautifulSoup**

LandingAI returns tables as HTML. BeautifulSoup parses the HTML to extract:
- Headers from `<th>` or first `<tr>` cells
- Data rows from subsequent `<tr>` elements

```python
# Input HTML from LandingAI
<table>
  <tr><th>Drug</th><th>Dose</th><th>Efficacy</th></tr>
  <tr><td>A</td><td>10mg</td><td>85%</td></tr>
  <tr><td>B</td><td>20mg</td><td>72%</td></tr>
</table>

# BeautifulSoup extracts:
ParsedTable(
    headers=["Drug", "Dose", "Efficacy"],
    rows=[["A", "10mg", "85%"], ["B", "20mg", "72%"]]
)
```

**Step 2: Row Context Formatting**

Each row is formatted as key-value pairs:
```
"Drug: A | Dose: 10mg | Efficacy: 85%"
```

**Step 3: Claude LLM Generates Declarative Statements**

Claude Opus 4.5 converts the formatted context into natural language:

```python
# Prompt to Claude:
"Convert the following table row data into a clear, factual natural language statement."

# Input: "Drug: A | Dose: 10mg | Efficacy: 85%"
# Output: "Drug A at a dose of 10mg demonstrated 85% efficacy."
```

**Final Output (linearized segments):**
```python
Segment(
    text="Drug A at a dose of 10mg demonstrated 85% efficacy.",
    segment_type="table_row",
    metadata={"source": "Table 1; row 1"}
)
Segment(
    text="Drug B at a dose of 20mg demonstrated 72% efficacy.",
    segment_type="table_row",
    metadata={"source": "Table 1; row 2"}
)
```

### 2.3 Figure Chunks

Figure descriptions from LandingAI are cleaned and transformed into declarative claims using Claude Opus 4.5.

**Input (LandingAI description):**
```
"Figure 3 shows a bar chart depicting symptom reduction. The treatment
arm (n=450) showed 68% reduction compared to 32% in placebo (n=448)."
```

**Output (cleaned claims):**
```python
Segment(
    text="The treatment arm (n=450) showed 68% symptom reduction.",
    segment_type="figure",
    metadata={"source": "Figure 3"}
)
Segment(
    text="The placebo arm (n=448) showed 32% symptom reduction.",
    segment_type="figure",
    metadata={"source": "Figure 3"}
)
```

---

## Step 3: Text Segmentation

Text segments are split into sentences using Stanza NLP (or gRPC sentenciser service). Table and figure segments are kept as single units.

| Segment Type | Segmentation |
|--------------|--------------|
| `text` | Split into sentences |
| `table_row`, `table_cell` | Kept as-is |
| `figure` | Kept as-is |

**Block-level segmentation:** When `segmentation_level="block"`, text segments are NOT split into sentences. The entire paragraph is indexed as one document.

**Minimum length filter:** Segments shorter than 5 characters are discarded.

---

## Step 4: Feature Extraction

For each text unit (sentence or block), the following features are extracted:

### 4.1 Vector Embedding

BioBERT generates a 768-dimensional dense vector for semantic similarity search.

```python
embedding = embedder.encode_single(text)  # Returns numpy array [768]
```

### 4.2 Numeric Tokens

Numbers are extracted using Stanza NER or regex patterns.

```python
# Input: "Patients showed 25% improvement over 12 weeks (p < 0.001)"
# Output: ["25", "12", "0.001"]
```

### 4.3 Named Entities

Biomedical entities are extracted using a trained NER model (en_ner_bc5cdr_md).

```python
# Input: "Cytisinicline binds to nicotinic acetylcholine receptors"
# Output:
#   entities: ["Cytisinicline", "nicotinic acetylcholine receptors"]
#   entity_types: ["CHEMICAL", "GENE"]
```

---

## Step 5: Document Assembly

All extracted data is assembled into an Elasticsearch document structure.

### Document Fields

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `ref_id` | string | PDF filename | Unique reference identifier |
| `ref_title` | string | Metadata or filename | Publication title |
| `sent_id` | string | Generated | Unique document ID: `{ref_id}::page{N}::para{N}::sent{N}` |
| `text` | string | Processed content | The actual indexed text |
| `section` | string | LandingAI | Document section (abstract, methods, etc.) |
| `page` | int | LandingAI | Page number (1-indexed) |
| `paragraph_number` | int | Counter | Paragraph position on page |
| `sentence_number` | int | Counter | Sentence position in paragraph |
| `bbox` | object | LandingAI | Bounding box coordinates {x0, y0, x1, y1} |
| `segment_type` | string | Processing | Content type: text, table_row, table_cell, figure |
| `source` | string | Processing | Source reference for tables/figures (null for text) |
| `vector` | float[768] | BioBERT | Dense embedding for semantic search |
| `numeric_tokens` | string[] | NLP | Extracted numbers |
| `entities` | string[] | NER | Named entities |
| `entity_types` | string[] | NER | Entity type labels |
| `labels` | string[] | Fixed | ["reference", {segment_type}] |
| `segmentation_strategy` | string | Fixed | Always "publication" |
| `segmentation_level` | string | Config | "sentence" or "block" |
| `org_id` | int | Input | Organization ID |
| `brand_id` | int | Input | Brand ID |
| `timestamp` | datetime | System | UTC timestamp of indexing |
| `doc_metadata` | object | LLM | Bibliographic metadata (title, authors, year, etc.) |

### doc_metadata Fields

Bibliographic metadata is extracted from the first page of the PDF using Claude LLM. The extractor first detects the document type, then extracts type-specific fields.

**Supported Document Types:**

| Type | Description |
|------|-------------|
| `journal_article` | Scientific/academic journal article with DOI, volume, issue |
| `book` | Complete book publication |
| `book_chapter` | Chapter within an edited book |
| `website` | Website or webpage |
| `online_video` | Online video content (YouTube, Vimeo, etc.) |
| `drug_database` | Drug/pharmaceutical database entry |

**Fields by Document Type:**

**journal_article** (most common):
| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Title of the article |
| `authors` | Yes | List of author names (e.g., `["Smith J", "Jones M"]`) |
| `journal_name` | Yes | Abbreviated journal name (e.g., `"N Engl J Med"`) |
| `year` | Yes | Publication year (4 digits) |
| `journal_name_full` | No | Full journal name |
| `volume` | No | Volume number |
| `issue` | No | Issue number |
| `supplement` | No | Supplement information |
| `page_range` | No | Page range (e.g., `"123-145"`) |
| `doi` | No | DOI number (e.g., `"10.1000/xyz123"`) |
| `publisher` | No | Publisher name |

**book**:
| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Book title |
| `authors` | Yes | List of author names |
| `publisher` | Yes | Publisher name |
| `year` | Yes | Publication year |
| `edition` | No | Edition number or name |
| `publication_city` | No | City of publication |
| `isbn` | No | ISBN number |

**book_chapter**:
| Field | Required | Description |
|-------|----------|-------------|
| `chapter_title` | Yes | Title of the chapter |
| `chapter_authors` | Yes | Authors of the chapter |
| `book_title` | Yes | Title of the containing book |
| `editors` | Yes | Book editor(s) |
| `publisher` | Yes | Publisher name |
| `year` | Yes | Publication year |
| `page_range` | No | Page range within the book |

**website**:
| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Page title |
| `website_name` | Yes | Name of the website (e.g., `"FDA.gov"`) |
| `url` | Yes | Full URL |
| `accessed_date` | Yes | Date accessed (YYYY-MM-DD) |
| `authors` | No | Author names if available |

### Assembled Document Example

```python
doc = {
    # Identifiers
    "ref_id": "Rigotti_JAMA_2023",
    "ref_title": "Smoking Cessation Trial Results",
    "sent_id": "Rigotti_JAMA_2023::page3::para2::sent1",

    # Content
    "text": "Patients showed a 25% reduction in symptoms (p < 0.001).",
    "section": "results",
    "page": 3,
    "paragraph_number": 2,
    "sentence_number": 1,

    # Spatial
    "bbox": {"x0": 72.0, "y0": 340.5, "x1": 540.0, "y1": 355.2},

    # Classification
    "segment_type": "text",
    "source": None,  # "Table 1; row 2" or "Figure 3" for tables/figures
    "labels": ["reference", "text"],
    "segmentation_strategy": "publication",
    "segmentation_level": "sentence",

    # NLP Features
    "vector": [0.023, -0.156, 0.089, ...],  # 768 floats
    "numeric_tokens": ["25", "0.001"],
    "entities": ["symptoms"],
    "entity_types": ["DISEASE"],

    # Metadata
    "org_id": 1,
    "brand_id": 1,
    "timestamp": "2025-03-14T10:30:45.123456",
    "doc_metadata": {
        "title": "Smoking Cessation Trial Results",
        "authors": ["Rigotti NA", "Chang Y"],
        "year": "2023",
        "journal_name": "JAMA",
        "doi": "10.1001/jama.2023.1234"
    }
}
```

---

## Step 6: Bulk Indexing to Elasticsearch

Documents are indexed in batches using Elasticsearch bulk API.

### Index Settings

```json
{
    "number_of_shards": 2,
    "number_of_replicas": 1
}
```

### Field Mappings

| Field | Elasticsearch Type | Purpose |
|-------|-------------------|---------|
| `ref_id` | `keyword` | Exact match filtering |
| `ref_title` | `text` | Full-text search |
| `sent_id` | `keyword` | Unique document ID |
| `text` | `text` | Full-text search |
| `section` | `keyword` | Section filtering |
| `page` | `integer` | Page filtering |
| `bbox` | `object` (floats) | Coordinate storage |
| `source` | `keyword` | Table/figure source filtering |
| `numeric_tokens` | `keyword[]` | Numeric matching |
| `entities` | `keyword[]` | Entity matching |
| `entity_types` | `keyword[]` | Entity type filtering |
| `vector` | `dense_vector(768)` | Cosine similarity search |
| `labels` | `keyword[]` | Label filtering |
| `org_id` | `integer` | Multi-tenancy filtering |
| `brand_id` | `integer` | Multi-tenancy filtering |
| `timestamp` | `date` | Time-based queries |

### Batch Processing

```python
# Default batch size: 500 documents
stats = bulk_index(es_client, index_name, documents, batch_size=500)
# Returns: {"success": 1234, "errors": 0}
```

---

## Content Type Examples

### Text Document

```json
{
  "ref_id": "Rigotti_JAMA_2023",
  "sent_id": "Rigotti_JAMA_2023::page3::para2::sent1",
  "text": "Cytisinicline, a partial agonist at α4β2 nicotinic acetylcholine receptors, has demonstrated smoking cessation efficacy.",
  "segment_type": "text",
  "source": null,
  "page": 3,
  "numeric_tokens": [],
  "entities": ["Cytisinicline", "nicotinic acetylcholine receptors"],
  "vector": [0.023, -0.156, ...]
}
```

### Table Document

```json
{
  "ref_id": "Rigotti_JAMA_2023",
  "sent_id": "Rigotti_JAMA_2023::page5::para1::sent1",
  "text": "In Table 2, row 3: Complete abstinence at 12 weeks was 28.5% for the treatment group.",
  "segment_type": "table_row",
  "source": "Table 2; row 3",
  "page": 5,
  "numeric_tokens": ["12", "28.5"],
  "entities": [],
  "vector": [0.045, -0.234, ...]
}
```

### Figure Document

```json
{
  "ref_id": "Rigotti_JAMA_2023",
  "sent_id": "Rigotti_JAMA_2023::page4::para1::sent1",
  "text": "The treatment arm (n=450) showed 68% symptom reduction over 12 weeks.",
  "segment_type": "figure",
  "source": "Figure 3",
  "page": 4,
  "numeric_tokens": ["450", "68", "12"],
  "entities": [],
  "vector": [0.067, -0.189, ...]
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_URL` | `http://localhost:9200` | Elasticsearch URL |
| `ES_INDEX` | `tf_index_med` | Index name |
| `EMBED_MODEL` | `pritamdeka/BioBERT-mnli-snli-...` | Embedding model |
| `NER_MODEL` | `en_ner_bc5cdr_md` | NER model |
| `NER_ENABLED` | `true` | Enable NER extraction |
| `FIGURE_ENABLED` | `true` | Enable figure processing |
| `FIGURE_MODEL` | `claude-opus-4-5-20251101` | Figure processing model |
| `INDEX_TEXT` | `true` | Index text by default |
| `INDEX_TABLES` | `true` | Index tables by default |
| `INDEX_FIGURES` | `true` | Index figures by default |

### CLI Options

```bash
revisto-index --refs-dir ./references --org-id 1 --brand-id 1 [OPTIONS]

Required:
  --refs-dir PATH       Directory containing reference PDFs
  --org-id INT          Organization ID
  --brand-id INT        Brand ID

Content Processing:
  --no-ner              Disable NER extraction
  --no-tables           Disable table linearization (skip Claude API calls)
  --no-figures          Disable figure processing (skip Claude API calls)

Content Type Indexing:
  --index-text / --no-index-text       Index or skip text content
  --index-tables / --no-index-tables   Index or skip table content
  --index-figures / --no-index-figures Index or skip figure content

Other:
  --segmentation-level  "sentence" (default) or "block"
  --batch-size INT      Bulk indexing batch size (default: 500)
  --log-level           DEBUG, INFO, WARNING, ERROR
```

### API Parameters

`POST /api/index/references`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | File[] | required | PDF files |
| `org_id` | int | required | Organization ID |
| `brand_id` | int | required | Brand ID |
| `enable_ner` | bool | true | Enable NER |
| `enable_tables` | bool | true | Enable table linearization |
| `enable_figures` | bool | true | Enable figure processing |
| `index_text` | bool | true | Index text content |
| `index_tables` | bool | true | Index table content |
| `index_figures` | bool | true | Index figure content |
| `segmentation_level` | string | "sentence" | Segmentation level |
| `batch_size` | int | 500 | Batch size |

---

## Related Files

| File | Purpose |
|------|---------|
| `core/indexing.py` | Main indexing pipeline |
| `utils/landingai_parser.py` | PDF parsing and chunk processing |
| `utils/table_linearizer.py` | Table to text conversion |
| `utils/figure_processor.py` | Figure claim extraction |
| `storage/elasticsearch.py` | ES client and operations |
| `storage/mappings.py` | Index mappings |
| `nlp/embeddings.py` | Embedding model |
| `nlp/ner.py` | NER extraction |
| `cli/index.py` | CLI entry point |
| `api/routers/indexing.py` | API endpoints |
