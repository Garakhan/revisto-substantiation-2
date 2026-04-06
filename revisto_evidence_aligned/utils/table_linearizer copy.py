"""
Table-to-Text linearization using Claude Opus.
Converts HTML tables to natural language statements for embedding.
"""

import anthropic
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from .secrets import get_anthropic_api_key
from .logging import get_logger

logger = get_logger(__name__)

# Maximum items per batch to avoid token limits
MAX_BATCH_SIZE = 20


@dataclass
class ParsedTable:
    """Parsed table structure."""
    headers: List[str]
    rows: List[List[str]]
    table_id: Optional[str] = None


class ClaudeLinearizer:
    """Linearize tables using Claude Opus."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5-20251101"):
        """
        Initialize the Claude linearizer.

        Args:
            api_key: Anthropic API key (defaults to secrets manager or env var)
            model: Claude model to use
        """
        self.api_key = api_key or get_anthropic_api_key()
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized Claude linearizer with model: {self.model}")

    def parse_html_table(self, html: str) -> ParsedTable:
        """
        Parse HTML table and extract headers and rows.

        Args:
            html: HTML string containing a table

        Returns:
            ParsedTable with headers and rows
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        if not table:
            raise ValueError("No table found in HTML")

        table_id = table.get("id")
        rows = table.find_all("tr")

        if not rows:
            return ParsedTable(headers=[], rows=[], table_id=table_id)

        # First row is header - check for th or td
        header_row = rows[0]
        header_cells = header_row.find_all("th") or header_row.find_all("td")
        headers = [cell.get_text(strip=True) for cell in header_cells]

        # Remaining rows are data
        data_rows = []
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if cells:
                data_rows.append(cells)

        return ParsedTable(headers=headers, rows=data_rows, table_id=table_id)

    def _format_row_context(self, headers: List[str], row: List[str]) -> str:
        """Format a row as context for Claude."""
        pairs = [f"{h}: {v}" for h, v in zip(headers, row)]
        return " | ".join(pairs)

    def generate_text(self, context: str, prompt_type: str = "row") -> str:
        """
        Generate natural language from table data using Claude.

        Args:
            context: Formatted table context
            prompt_type: Type of prompt ('row' or 'cell')

        Returns:
            Generated natural language text
        """
        if prompt_type == "row":
            prompt = f"""Convert the following table row data into a clear, factual natural language statement.
Be concise and focus on the key information. Output only the statement, no explanation.

Table row data: {context}

Natural language statement:"""
        else:
            prompt = f"""Convert the following table cell data into a brief, factual natural language statement.
Be concise and specific. Output only the statement, no explanation.

Cell data: {context}

Natural language statement:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.content:
                logger.warning(f"Empty response from Claude API for context: {context[:100]}...")
                return ""

            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""

    def generate_text_batch(self, contexts: List[Dict[str, str]], prompt_type: str = "row") -> List[str]:
        """
        Generate natural language for multiple items in a single API call.

        Args:
            contexts: List of dicts with 'id' and 'context' keys
            prompt_type: Type of prompt ('row' or 'cell')

        Returns:
            List of generated texts in the same order as inputs
        """
        if not contexts:
            return []

        if prompt_type == "row":
            instruction = "Convert each table row into a clear, factual natural language statement."
        else:
            instruction = "Convert each table cell into a brief, factual natural language statement."

        # Build batch prompt
        items_text = "\n".join([
            f"[{item['id']}] {item['context']}"
            for item in contexts
        ])

        prompt = f"""{instruction}
Be concise and focus on key information. Output ONLY a JSON array of objects with "id" and "text" fields.

Items to convert:
{items_text}

Output format: [{{"id": "1", "text": "statement"}}, ...]
JSON output:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.content:
                logger.warning("Empty response from Claude API for batch generation")
                return [""] * len(contexts)

            # Parse JSON response
            response_text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            results = json.loads(response_text)

            # Map results back by ID
            result_map = {str(r["id"]): r["text"] for r in results}
            return [result_map.get(item["id"], "") for item in contexts]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch response as JSON: {e}")
            # Fall back to individual calls
            return [self.generate_text(item["context"], prompt_type) for item in contexts]
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return [""] * len(contexts)

    def linearize_by_row(self, parsed_table: ParsedTable) -> List[Dict]:
        """
        Generate one statement per row using batched API calls.

        Args:
            parsed_table: Parsed table data

        Returns:
            List of dicts with row_entity and generated_text
        """
        # Prepare batch contexts
        batch_items = []
        row_metadata = []  # Store row info for results

        for idx, row in enumerate(parsed_table.rows):
            if not row:
                continue
            context = self._format_row_context(parsed_table.headers, row)
            batch_items.append({
                "id": str(idx),
                "context": context
            })
            row_metadata.append({
                "row_entity": row[0] if row else "",
                "input": context
            })

        if not batch_items:
            logger.info("No rows to linearize — skipping")
            return []

        logger.info(f"Linearizing {len(batch_items)} rows via Claude API (batch size {MAX_BATCH_SIZE})")

        # Process in batches
        all_generated = []
        for i in range(0, len(batch_items), MAX_BATCH_SIZE):
            batch = batch_items[i:i + MAX_BATCH_SIZE]
            batch_num = i // MAX_BATCH_SIZE + 1
            total_batches = (len(batch_items) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            logger.info(f"  Row batch {batch_num}/{total_batches}: sending {len(batch)} rows to Claude API")
            generated = self.generate_text_batch(batch, prompt_type="row")
            logger.info(f"  Row batch {batch_num}/{total_batches}: received {sum(1 for g in generated if g)} results")
            all_generated.extend(generated)

        # Combine results
        results = []
        for meta, generated in zip(row_metadata, all_generated):
            results.append({
                "row_entity": meta["row_entity"],
                "input": meta["input"],
                "generated_text": generated
            })

        logger.info(f"Row linearization complete: {len(results)} statements generated")
        return results

    def linearize_by_cell(self, parsed_table: ParsedTable) -> List[Dict]:
        """
        Generate one statement per cell using batched API calls.

        Args:
            parsed_table: Parsed table data

        Returns:
            List of dicts with cell info and generated_text
        """
        # Prepare batch contexts
        batch_items = []
        cell_metadata = []  # Store cell info for results

        idx = 0
        for row in parsed_table.rows:
            if not row:
                continue
            row_entity = row[0] if row else ""

            for i, value in enumerate(row[1:], start=1):
                if i >= len(parsed_table.headers):
                    continue
                header = parsed_table.headers[i]
                context = f"{row_entity} - {header}: {value}"

                batch_items.append({
                    "id": str(idx),
                    "context": context
                })
                cell_metadata.append({
                    "row_entity": row_entity,
                    "column": header,
                    "value": value,
                    "input": context
                })
                idx += 1

        if not batch_items:
            logger.info("No cells to linearize — skipping")
            return []

        logger.info(f"Linearizing {len(batch_items)} cells via Claude API (batch size {MAX_BATCH_SIZE})")

        # Process in batches
        all_generated = []
        for i in range(0, len(batch_items), MAX_BATCH_SIZE):
            batch = batch_items[i:i + MAX_BATCH_SIZE]
            batch_num = i // MAX_BATCH_SIZE + 1
            total_batches = (len(batch_items) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            logger.info(f"  Cell batch {batch_num}/{total_batches}: sending {len(batch)} cells to Claude API")
            generated = self.generate_text_batch(batch, prompt_type="cell")
            logger.info(f"  Cell batch {batch_num}/{total_batches}: received {sum(1 for g in generated if g)} results")
            all_generated.extend(generated)

        # Combine results
        results = []
        for meta, generated in zip(cell_metadata, all_generated):
            results.append({
                "row_entity": meta["row_entity"],
                "column": meta["column"],
                "value": meta["value"],
                "input": meta["input"],
                "generated_text": generated
            })

        logger.info(f"Cell linearization complete: {len(results)} statements generated")
        return results

    def linearize_table_element(
        self,
        table_element: Dict,
        mode: str = "cell"
    ) -> List[Dict]:
        """
        Process a table element from document extraction.

        Args:
            table_element: Dict with 'markdown' key containing HTML table
            mode: 'cell' for per-cell statements, 'row' for per-row

        Returns:
            List of linearized statements
        """
        html = table_element.get("markdown", "")
        if not html:
            logger.info("Table element has no HTML — skipping")
            return []

        table_id = table_element.get("id", "unknown")
        page = table_element.get("grounding", {}).get("page", "?")
        logger.info(f"Processing table (id={table_id}, page={page}, mode={mode})")

        try:
            parsed = self.parse_html_table(html)
        except ValueError as e:
            logger.warning(f"Failed to parse table (id={table_id}): {e}")
            return []

        if not parsed.rows:
            logger.info(f"Table (id={table_id}) has no data rows — skipping")
            return []

        logger.info(f"Table (id={table_id}): {len(parsed.headers)} columns, {len(parsed.rows)} rows")

        if mode == "cell":
            results = self.linearize_by_cell(parsed)
        else:
            results = self.linearize_by_row(parsed)

        # Add table metadata
        for r in results:
            r["table_id"] = table_element.get("id")
            r["page"] = page

        logger.info(f"Table (id={table_id}) done: {len(results)} statements")
        return results
