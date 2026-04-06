# Table Interpretation Flow

## Overview

This document describes how tables are extracted from PDFs, linearized into natural language, and indexed for semantic search.

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Input
        PDF[/"PDF Document"/]
    end

    subgraph Parsing["Document Parsing"]
        PDF --> |"LandingAI API"| Parse[Parse PDF]
        Parse --> |"JSON response"| Chunks[Chunks Array]
        Chunks --> TextChunk["type: text"]
        Chunks --> TableChunk["type: table"]
    end

    subgraph TextFlow["Text Processing"]
        TextChunk --> |"PyMuPDF<br/>fitz"| TextCoords[Extract Coordinates]
        TextCoords --> |"regex"| CleanText[Clean HTML tags]
        CleanText --> TextSegment["Segment<br/>segment_type: text<br/>table_source: null"]
    end

    subgraph TableFlow["Table Processing"]
        TableChunk --> |"BeautifulSoup"| ParseHTML[Parse HTML Table]
        ParseHTML --> ParsedTable["ParsedTable<br/>headers + rows"]

        ParsedTable --> RowLin[linearize_by_row]
        ParsedTable --> CellLin[linearize_by_cell]

        subgraph Claude["Claude Opus API"]
            RowLin --> |"anthropic SDK"| RowPrompt["Prompt: Convert row data<br/>to natural language"]
            CellLin --> |"anthropic SDK"| CellPrompt["Prompt: Convert cell data<br/>to natural language"]
        end

        RowPrompt --> RowSeg["Segment<br/>segment_type: table_row<br/>table_source: Table 1; row 1"]
        CellPrompt --> CellSeg["Segment<br/>segment_type: table_cell<br/>table_source: Table 1; row 1; column 1"]
    end

    subgraph Embedding["Embedding & NLP"]
        TextSegment --> |"Stanza gRPC"| Sentencize[Sentencize]
        RowSeg --> Skip1[Skip sentencization]
        CellSeg --> Skip2[Skip sentencization]

        Sentencize --> |"BioBERT gRPC"| Embed1[Generate Embedding]
        Skip1 --> |"BioBERT gRPC"| Embed2[Generate Embedding]
        Skip2 --> |"BioBERT gRPC"| Embed3[Generate Embedding]

        Embed1 --> |"BioNER gRPC"| NER1[Extract Entities]
        Embed2 --> |"BioNER gRPC"| NER2[Extract Entities]
        Embed3 --> |"BioNER gRPC"| NER3[Extract Entities]
    end

    subgraph Indexing["Indexing"]
        NER1 --> |"elasticsearch-py"| ES[(Elasticsearch)]
        NER2 --> |"elasticsearch-py"| ES
        NER3 --> |"elasticsearch-py"| ES
    end

    subgraph Search["Search API"]
        Claim[/"Claim Text"/] --> |"FastAPI"| API["/api/search/claim"]
        API --> |"BioBERT"| ClaimEmbed[Embed Claim]
        ClaimEmbed --> |"script_score query"| ES
        ES --> |"JSON"| Results[/"Results with<br/>table_source"/]
    end

    style Claude fill:#f9f,stroke:#333
    style ES fill:#9cf,stroke:#333
```

## Table Linearization Example

```mermaid
flowchart LR
    subgraph Table["HTML Table"]
        T["| Adverse Event | Drug A | Placebo |<br/>| Headache | 7.6% | 7.1% |<br/>| Nausea | 9.4% | 4.7% |"]
    end

    subgraph RowOutput["Row Linearization"]
        R1["Row 1: Headache occurred in 7.6%<br/>of Drug A patients vs 7.1% placebo"]
        R2["Row 2: Nausea occurred in 9.4%<br/>of Drug A patients vs 4.7% placebo"]
    end

    subgraph CellOutput["Cell Linearization"]
        C1["Row 1, Col 1: Headache occurred<br/>in 7.6% of Drug A patients"]
        C2["Row 1, Col 2: Headache occurred<br/>in 7.1% of placebo patients"]
        C3["Row 2, Col 1: Nausea occurred<br/>in 9.4% of Drug A patients"]
        C4["Row 2, Col 2: Nausea occurred<br/>in 4.7% of placebo patients"]
    end

    Table --> RowOutput
    Table --> CellOutput
```

## Technology Stack

| Step | Tool | Description |
|------|------|-------------|
| PDF Parsing | LandingAI API | Extracts text and table chunks from PDFs |
| HTML Table Parsing | BeautifulSoup | Parses HTML table structure into headers and rows |
| Table Linearization | Claude Opus | Converts table data to natural language statements |
| Sentence Splitting | Stanza (gRPC) | Splits text into sentences |
| Embeddings | BioBERT (gRPC) | Generates semantic embeddings |
| Entity Extraction | BioNER (gRPC) | Extracts biomedical entities |
| Storage & Search | Elasticsearch | Vector search with script_score |
| API | FastAPI | REST API endpoints |

## Data Flow

1. **PDF Input**: Document is uploaded to the indexing endpoint
2. **LandingAI Parsing**: PDF is parsed into chunks (text and tables)
3. **Text Processing**: Text chunks are cleaned and segmented
4. **Table Processing**:
   - HTML tables are parsed with BeautifulSoup
   - Claude generates both row-level and cell-level natural language statements
   - Each statement becomes a separate segment
5. **Embedding**: All segments are embedded using BioBERT
6. **Indexing**: Segments are indexed in Elasticsearch with:
   - `segment_type`: "text", "table_row", or "table_cell"
   - `table_source`: Location reference (e.g., "Table 1; row 1; column 1") or null for text
7. **Search**: Claims are embedded and matched against indexed segments

## Output Fields

```json
{
  "text": "Headache occurred in 7.6% of patients taking Drug A.",
  "segment_type": "table_cell",
  "table_source": "Table 1; row 1; column 1",
  "page": 6,
  "relevance_score": 0.85
}
```
