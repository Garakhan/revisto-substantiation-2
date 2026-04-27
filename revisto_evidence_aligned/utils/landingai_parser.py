import re
import json
import httpx
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from revisto_evidence_aligned.utils.coord_conversion import landingai_to_pymupdf_coordinates
from revisto_evidence_aligned.utils.types import Segment
from revisto_evidence_aligned.utils.metadata_extractor import MetadataExtractor
from revisto_evidence_aligned.utils.secrets import (
    get_landingai_api_key,
    get_landingai_output_bucket,
    generate_presigned_upload_url,
    generate_presigned_download_url,
    upload_to_s3,
    download_from_s3
)

OUTPUT_TYPE = ["markdown", "chunks", "splits", "grounding", "metadata"]

LANDINGAI_TIMEOUT = 1800.0  # 30 minutes

async def process_pdf_landingai(pdf_path: str, extract_metadata: bool = True, cache_dir: str = None):
    """Parse a PDF using LandingAI and optionally extract metadata.

    Args:
        pdf_path: Path to the PDF file.
        extract_metadata: Whether to extract bibliographic metadata using LLM.
        cache_dir: Directory to save cached JSON results. If None, saves next to PDF.

    Returns:
        Tuple of (chunks, metadata) if extract_metadata is True, else just chunks.
    """
    # Determine cache path - check multiple locations
    pdf_name = Path(pdf_path).stem
    json_path = None

    # List of possible cache locations to check
    cache_locations = []
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_locations.append(cache_dir_path / f"{pdf_name}.json")

    # Also check next to the PDF file
    cache_locations.append(Path(pdf_path).with_suffix(".json"))

    # Also check in the PDF's parent directory
    cache_locations.append(Path(pdf_path).parent / f"{pdf_name}.json")

    # Find first existing cache file
    for loc in cache_locations:
        if loc.exists():
            json_path = loc
            print(f"Found cached JSON at: {json_path}")
            break

    # If no cache found, use the primary cache location for saving
    if json_path is None:
        json_path = cache_locations[0] if cache_locations else Path(pdf_path).with_suffix(".json")
        print(f"No cache found for {pdf_name}, will process and save to: {json_path}")

    if json_path.exists():
        print(f"Loading cached parsing results from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", data if isinstance(data, list) else [])
        markdown = data.get("markdown") if isinstance(data, dict) else None
        metadata = data.get("metadata") if isinstance(data, dict) else None

        # Reconstruct markdown from chunks if missing (for old cache files)
        cache_updated = False
        if not markdown and chunks:
            print("Reconstructing markdown from chunks...")
            markdown = "\n\n".join(
                chunk.get("markdown", chunk.get("text", ""))
                for chunk in chunks
                if chunk.get("type") == "text"
            )
            # Update cache with reconstructed markdown
            if isinstance(data, dict):
                data["markdown"] = markdown
                cache_updated = True

        # If we have cached chunks but no metadata and extraction is requested, extract it now
        if extract_metadata and chunks and not metadata and markdown:
            print("Extracting metadata from cached markdown...")
            extractor = MetadataExtractor()
            metadata = await extractor.extract(markdown, max_words=300)
            # Update cache with metadata
            data["metadata"] = metadata
            cache_updated = True

        # Save updated cache if anything changed
        if cache_updated:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Updated cache at {json_path}")

        if extract_metadata:
            return chunks, metadata
        return chunks

    api_key = get_landingai_api_key()
    if not api_key:
        raise RuntimeError("LandingAI API key not configured - check AWS Secrets Manager or LANDING_AI_API_KEY env var")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    client = LandingAIParser(headers=headers)

    # Check if S3 bucket is configured for zero data retention mode
    s3_bucket = get_landingai_output_bucket()
    print(f"DEBUG: LANDINGAI_OUTPUT_BUCKET = {s3_bucket}")
    s3_output_key = None
    output_save_url = None
    document_url = None

    if s3_bucket:
        # For zero data retention: upload PDF to S3 and use URLs instead of file upload
        unique_id = uuid.uuid4().hex[:8]

        # Upload PDF to S3
        s3_input_key = f"landingai-input/{pdf_name}_{unique_id}.pdf"
        print(f"Uploading PDF to S3: s3://{s3_bucket}/{s3_input_key}")
        upload_to_s3(s3_bucket, s3_input_key, pdf_path)

        # Generate presigned URL for LandingAI to download the PDF
        document_url = generate_presigned_download_url(s3_bucket, s3_input_key, expiration=7200)

        # Generate presigned URL for LandingAI to upload results
        s3_output_key = f"landingai-output/{pdf_name}_{unique_id}.json"
        output_save_url = generate_presigned_upload_url(s3_bucket, s3_output_key, expiration=7200)

        print(f"Using S3 for zero data retention mode")
        print(f"DEBUG: output_save_url = {output_save_url}")
        print(f"  Input: s3://{s3_bucket}/{s3_input_key}")
        print(f"  Output: s3://{s3_bucket}/{s3_output_key}")

    # Step 1: Create job
    job_response = await client.create_job(
        pdf_path,
        document_url=document_url,
        output_save_url=output_save_url
    )
    print(f"DEBUG: job_response = {job_response}")
    job_id = job_response.get("job_id")

    if not job_id:
        raise RuntimeError("No job_id returned")

    # Step 2: Wait for job to complete
    final_status_data = None
    while True:
        status_data = await client.monitor_job_status(job_id)
        if status_data and status_data.get("status") == "completed":
            final_status_data = status_data
            print(f"DEBUG: Final status data: {status_data}")
            break
        await asyncio.sleep(5)

    # Step 3: Retrieve results (chunks and markdown)
    if s3_bucket and s3_output_key:
        # Download from S3 (zero data retention mode)
        # Retry a few times as there might be a delay after job completion
        max_retries = 5
        retry_delay = 3
        s3_content = None

        for attempt in range(max_retries):
            try:
                print(f"Downloading results from S3 (attempt {attempt + 1}/{max_retries}): s3://{s3_bucket}/{s3_output_key}")
                s3_content = download_from_s3(s3_bucket, s3_output_key)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"S3 download failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Error downloading from S3 after {max_retries} attempts: {e}")

        if s3_content:
            result_data = json.loads(s3_content)
            chunks = result_data.get("chunks")
            markdown = result_data.get("markdown")
        else:
            chunks = None
            markdown = None
    else:
        # Standard mode - retrieve from LandingAI API
        result_data = await client.retrieve_full_results(job_id, pdf_path=pdf_path)
        chunks = result_data.get("chunks") if result_data else None
        markdown = result_data.get("markdown") if result_data else None

    # Step 4: Extract metadata if requested (using markdown, not chunks)
    metadata = None
    if extract_metadata and markdown:
        print("Extracting document metadata...")
        extractor = MetadataExtractor()
        metadata = await extractor.extract(markdown, max_words=300)
        print(f"Metadata extraction complete: {metadata.get('document_type_info', {}).get('document_type', 'unknown')}")

    # Save results to cache (include chunks, markdown, and metadata)
    if chunks:
        cache_data = {"chunks": chunks}
        if markdown:
            cache_data["markdown"] = markdown
        if metadata:
            cache_data["metadata"] = metadata
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"Saved parsing results to {json_path}")

    if extract_metadata:
        return chunks, metadata
    return chunks

def _process_text_chunks(
    text_chunks: List[Dict[str, Any]],
    doc
) -> List[Segment]:
    """Process text chunks into segments.

    Args:
        text_chunks: List of text chunks from LandingAI.
        doc: PyMuPDF document object.

    Returns:
        List of Segment objects.
    """
    segments = []

    for chunk in text_chunks:
        chunk_page_number = chunk.get("grounding", {}).get("page", 0)
        chunk_box_raw = chunk.get("grounding", {}).get("box")

        if chunk_box_raw:
            chunk_box = landingai_to_pymupdf_coordinates(chunk_box_raw, doc[chunk_page_number])
        else:
            chunk_box = (0, 0, 0, 0)

        chunk_text = re.sub(r"<a.*?</a>", "", chunk.get("markdown", "")).strip()
        if chunk_text:
            segments.append(Segment(
                text=chunk_text,
                page=chunk_page_number,
                bbox=chunk_box,
                segment_type="text",
                metadata={
                    "source": None
                }
            ))

    return segments


def _process_single_table(table_linearizer, parsed_table, chunk_page_number, chunk_box, table_num):
    """Process a single table concurrently: rows and cells in parallel."""
    segments = []

    # Run row and cell linearization concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        row_future = executor.submit(table_linearizer.linearize_by_row, parsed_table)
        cell_future = executor.submit(table_linearizer.linearize_by_cell, parsed_table)

        row_results = row_future.result()
        cell_results = cell_future.result()

    # Build row segments
    for row_idx, result in enumerate(row_results, start=1):
        generated_text = result.get("generated_text", "")
        if generated_text:
            source = f"Table {table_num}; row {row_idx}"
            segments.append(Segment(
                text=generated_text,
                page=chunk_page_number,
                bbox=chunk_box,
                segment_type="table_row",
                metadata={"source": source}
            ))

    # Build cell segments
    current_row_entity = None
    row_num = 0
    col_num = 0
    for result in cell_results:
        generated_text = result.get("generated_text", "")
        if not generated_text:
            continue
        if result.get("row_entity") != current_row_entity:
            current_row_entity = result.get("row_entity")
            row_num += 1
            col_num = 0
        col_num += 1
        source = f"Table {table_num}; row {row_num}; column {col_num}"
        segments.append(Segment(
            text=generated_text,
            page=chunk_page_number,
            bbox=chunk_box,
            segment_type="table_cell",
            metadata={"source": source}
        ))

    return segments


def _process_table_chunks(
    table_chunks: List[Dict[str, Any]],
    doc,
    table_linearizer,
    segmentation_level: str = "sentence"
) -> List[Segment]:
    """Process table chunks into segments with concurrent table processing.

    Args:
        table_chunks: List of table chunks from LandingAI.
        doc: PyMuPDF document object.
        table_linearizer: ClaudeLinearizer instance for table processing.
        segmentation_level: "sentence" for cell-by-cell, "block" for row-by-row tables.

    Returns:
        List of Segment objects.
    """
    if not table_linearizer:
        return []

    # Pre-parse all tables and assign table numbers (must be sequential for naming)
    table_jobs = []
    table_count_per_page = {}

    for chunk in table_chunks:
        chunk_page_number = chunk.get("grounding", {}).get("page", 0)
        chunk_box_raw = chunk.get("grounding", {}).get("box")

        if chunk_box_raw:
            chunk_box = landingai_to_pymupdf_coordinates(chunk_box_raw, doc[chunk_page_number])
        else:
            chunk_box = (0, 0, 0, 0)

        table_html = chunk.get("markdown", "")
        if not table_html:
            continue

        try:
            parsed_table = table_linearizer.parse_html_table(table_html)
            if not parsed_table.rows:
                continue

            if chunk_page_number not in table_count_per_page:
                table_count_per_page[chunk_page_number] = 0
            table_count_per_page[chunk_page_number] += 1
            table_num = table_count_per_page[chunk_page_number]

            table_jobs.append((parsed_table, chunk_page_number, chunk_box, table_num))
        except Exception as e:
            print(f"Error parsing table: {e}")
            continue

    if not table_jobs:
        return []

    # Process all tables concurrently (max 5 tables at once to limit API pressure)
    all_segments = [None] * len(table_jobs)

    with ThreadPoolExecutor(max_workers=min(len(table_jobs), 5)) as executor:
        futures = {}
        for idx, (parsed_table, page_num, box, table_num) in enumerate(table_jobs):
            future = executor.submit(
                _process_single_table, table_linearizer, parsed_table, page_num, box, table_num
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                all_segments[idx] = future.result()
            except Exception as e:
                print(f"Error linearizing table {idx}: {e}")
                all_segments[idx] = []

    # Flatten in order
    segments = []
    for table_segs in all_segments:
        segments.extend(table_segs)

    return segments


def _process_figure_chunks(
    figure_chunks: List[Dict[str, Any]],
    doc,
    figure_processor
) -> List[Segment]:
    """Process figure chunks into segments.

    Args:
        figure_chunks: List of figure chunks from LandingAI.
        doc: PyMuPDF document object.
        figure_processor: FigureProcessor instance for description cleaning.

    Returns:
        List of Segment objects.
    """
    if not figure_processor:
        return []

    # Collect figure jobs (description + metadata, no numbering yet)
    figure_jobs = []

    for chunk in figure_chunks:
        chunk_page_number = chunk.get("grounding", {}).get("page", 0)
        chunk_box_raw = chunk.get("grounding", {}).get("box")

        if chunk_box_raw:
            chunk_box = landingai_to_pymupdf_coordinates(chunk_box_raw, doc[chunk_page_number])
        else:
            chunk_box = (0, 0, 0, 0)

        description = chunk.get("markdown", "").strip()
        if not description:
            continue

        figure_jobs.append((description, chunk_page_number, chunk_box))

    if not figure_jobs:
        return []

    # Process all figures concurrently — get claims only
    all_results = [None] * len(figure_jobs)

    def _process_single_figure(idx, description):
        try:
            claims = figure_processor.process(description)
            return idx, claims or []
        except Exception as e:
            print(f"Error processing figure: {e}")
            return idx, []

    with ThreadPoolExecutor(max_workers=min(len(figure_jobs), 5)) as executor:
        futures = [
            executor.submit(_process_single_figure, idx, desc)
            for idx, (desc, _, _) in enumerate(figure_jobs)
        ]
        for future in as_completed(futures):
            idx, claims = future.result()
            all_results[idx] = claims

    # Assign figure numbers sequentially (only for figures with claims, matching original logic)
    segments = []
    figure_count_per_page = {}

    for idx, (description, chunk_page_number, chunk_box) in enumerate(figure_jobs):
        claims = all_results[idx]
        if not claims:
            continue

        # Increment figure count only for successful figures (same as original)
        if chunk_page_number not in figure_count_per_page:
            figure_count_per_page[chunk_page_number] = 0
        figure_count_per_page[chunk_page_number] += 1
        figure_num = figure_count_per_page[chunk_page_number]

        source = f"Figure {figure_num}"
        for claim in claims:
            if not claim.strip():
                continue
            segments.append(Segment(
                text=claim,
                page=chunk_page_number,
                bbox=chunk_box,
                segment_type="figure",
                metadata={"source": source}
            ))

    return segments


def process_landingai_chunks(
    pdf_path: str,
    chunks: list,
    table_linearizer=None,
    figure_processor=None,
    segmentation_level: str = "sentence",
    index_text: bool = True,
    index_tables: bool = True,
    index_figures: bool = True
) -> List[Segment]:
    """Process LandingAI chunks into segments.

    Args:
        pdf_path: Path to the PDF file.
        chunks: List of chunks from LandingAI.
        table_linearizer: Optional ClaudeLinearizer instance for table processing.
        figure_processor: Optional FigureProcessor instance for figure processing.
        segmentation_level: "sentence" for cell-by-cell, "block" for row-by-row tables.
        index_text: Whether to index text chunks.
        index_tables: Whether to index table chunks.
        index_figures: Whether to index figure chunks.

    Returns:
        List of Segment objects.
    """
    import fitz  # Lazy import - only needed for PDF processing
    doc = fitz.open(pdf_path)

    segments = []

    # Process text chunks if enabled
    if index_text:
        text_chunks = [c for c in chunks if c.get("type") == "text"]
        segments.extend(_process_text_chunks(text_chunks, doc))

    # Process table chunks if enabled
    if index_tables:
        table_chunks = [c for c in chunks if c.get("type") == "table"]
        segments.extend(_process_table_chunks(table_chunks, doc, table_linearizer, segmentation_level))

    # Process figure chunks if enabled
    if index_figures:
        figure_chunks = [c for c in chunks if c.get("type") == "figure"]
        segments.extend(_process_figure_chunks(figure_chunks, doc, figure_processor))

    doc.close()

    return segments


async def process_landingai_chunks_async(
    pdf_path: str,
    chunks: list,
    table_linearizer=None,
    figure_processor=None,
    segmentation_level: str = "sentence",
    index_text: bool = True,
    index_tables: bool = True,
    index_figures: bool = True
) -> List[Segment]:
    """Process LandingAI chunks into segments with parallel processing.

    Text, table, and figure chunks are processed concurrently.

    Args:
        pdf_path: Path to the PDF file.
        chunks: List of chunks from LandingAI.
        table_linearizer: Optional ClaudeLinearizer instance for table processing.
        figure_processor: Optional FigureProcessor instance for figure processing.
        segmentation_level: "sentence" for cell-by-cell, "block" for row-by-row tables.
        index_text: Whether to index text chunks.
        index_tables: Whether to index table chunks.
        index_figures: Whether to index figure chunks.

    Returns:
        List of Segment objects.
    """
    import fitz  # Lazy import - only needed for PDF processing
    doc = fitz.open(pdf_path)

    # Collect futures for enabled content types
    futures = []
    future_types = []  # Track which type each future corresponds to

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Process text chunks if enabled
        if index_text:
            text_chunks = [c for c in chunks if c.get("type") == "text"]
            futures.append(loop.run_in_executor(
                executor,
                _process_text_chunks,
                text_chunks,
                doc
            ))
            future_types.append("text")

        # Process table chunks if enabled
        if index_tables:
            table_chunks = [c for c in chunks if c.get("type") == "table"]
            futures.append(loop.run_in_executor(
                executor,
                _process_table_chunks,
                table_chunks,
                doc,
                table_linearizer,
                segmentation_level
            ))
            future_types.append("table")

        # Process figure chunks if enabled
        if index_figures:
            figure_chunks = [c for c in chunks if c.get("type") == "figure"]
            futures.append(loop.run_in_executor(
                executor,
                _process_figure_chunks,
                figure_chunks,
                doc,
                figure_processor
            ))
            future_types.append("figure")

        # Wait for all enabled futures to complete
        if futures:
            results = await asyncio.gather(*futures)
        else:
            results = []

    doc.close()

    # Combine all segments
    all_segments = []
    for result in results:
        all_segments.extend(result)

    return all_segments

async def download_and_load_json(pdf_path: str, output_url: str) -> dict:
    # Change .pdf to .json
    json_path = Path(pdf_path).with_suffix(".json")

    # Download and save JSON
    async with httpx.AsyncClient(timeout=LANDINGAI_TIMEOUT) as client:
        async with client.stream("GET", output_url) as resp:
            resp.raise_for_status()
            with open(json_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)

    # Load JSON into memory
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data  # This is your usable parsed result (e.g., for chunking)


class LandingAIParser:
    def __init__(self, headers: dict):
        self.headers = headers
        self.base_url = "https://api.va.landing.ai/v1/ade/parse/jobs"

    async def create_job(self, pdf_path: str, document_url: str = None, output_save_url: str = None):
        url = self.base_url

        async with httpx.AsyncClient(timeout=LANDINGAI_TIMEOUT) as client:
            if document_url:
                # Zero data retention mode: use multipart form data with URLs
                # Use files dict with None for file content to force multipart encoding
                files = {
                    "document_url": (None, document_url),
                    "output_save_url": (None, output_save_url)
                }
                print(f"DEBUG create_job (URL mode) v3")
                print(f"DEBUG: document_url={document_url}")
                print(f"DEBUG: output_save_url={output_save_url}")
                response = await client.post(url, files=files, headers=self.headers)
            else:
                # Standard mode: upload file directly
                with open(pdf_path, "rb") as f:
                    files = {"document": (pdf_path, f, "application/pdf")}
                    response = await client.post(url, files=files, headers=self.headers)

        # Log response for debugging
        if response.status_code != 200:
            print(f"LandingAI API error: status={response.status_code}, body={response.text[:500]}")

        result = response.json()
        if "job_id" not in result:
            print(f"LandingAI API response missing job_id: {result}")

        return result

    async def monitor_job_status(self, job_id: int):
        job_url = f"{self.base_url}/{job_id}"

        async with httpx.AsyncClient(timeout=LANDINGAI_TIMEOUT) as client:
            response = await client.get(job_url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            progress = data.get("progress", 0) * 100
            print(f"Status: {status} | Progress: {progress:.0f}%")
            return data
        else:
            print(f"Error checking status: {response.status_code}")
            return None

    async def retrieve_results(self, job_id: int, output_type: str, pdf_path: str):
        url = f"{self.base_url}/{job_id}"

        async with httpx.AsyncClient(timeout=LANDINGAI_TIMEOUT) as client:
            response = await client.get(url, headers=self.headers)

        response_data = response.json()

        if response_data.get("status") == "completed":
            if response_data.get("data"):
                return response_data["data"][output_type]

            elif response_data.get("output_url"):
                download_url = response_data["output_url"]

                # Stream the file to disk
                output_path = str(Path(pdf_path).with_suffix(".json"))
                data = await download_and_load_json(output_path, download_url)
                return data.get("chunks")
            else:
                print("No Markdown content or output_url found.")
                return None
        else:
            print(f"Job status: {response_data.get('status', 'unknown')}.")
            return None

    async def retrieve_full_results(self, job_id: int, pdf_path: str) -> dict:
        """Retrieve both chunks and markdown from a completed job."""
        url = f"{self.base_url}/{job_id}"

        async with httpx.AsyncClient(timeout=LANDINGAI_TIMEOUT) as client:
            response = await client.get(url, headers=self.headers)

        response_data = response.json()

        if response_data.get("status") == "completed":
            if response_data.get("data"):
                return {
                    "chunks": response_data["data"].get("chunks"),
                    "markdown": response_data["data"].get("markdown")
                }

            elif response_data.get("output_url"):
                download_url = response_data["output_url"]
                output_path = str(Path(pdf_path).with_suffix(".json"))
                data = await download_and_load_json(output_path, download_url)
                return {
                    "chunks": data.get("chunks"),
                    "markdown": data.get("markdown")
                }
            else:
                print("No data or output_url found.")
                return None
        else:
            print(f"Job status: {response_data.get('status', 'unknown')}.")
            return None

