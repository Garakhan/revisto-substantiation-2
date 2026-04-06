"""
Extract images from a PDF using PyMuPDF and interpret each image using Claude.

Two-pass approach:
1. Classify images with Claude Haiku (fast, cheap) to identify charts/tables/graphs
2. Interpret only data visualizations with Claude Opus (detailed extraction)

Usage:
    python extract_and_interpret_images.py <pdf_path> [options]

Examples:
    # Basic usage - extract and interpret images, save results to JSON
    python extract_and_interpret_images.py document.pdf -o ./visual_claims

    # Save extracted images alongside JSON results
    python extract_and_interpret_images.py document.pdf -o ./visual_claims --save-images

    # For image-based PDFs, render full pages instead of extracting embedded images
    python extract_and_interpret_images.py scanned_doc.pdf -o ./visual_claims --render-pages --no-embedded

    # Custom prompt for specific data extraction
    python extract_and_interpret_images.py clinical.pdf -o ./visual_claims -p "Extract all numerical values and statistics"

    # Skip classification and interpret all images (slower, more expensive)
    python extract_and_interpret_images.py doc.pdf -o ./visual_claims --no-classify

    # Higher quality rendering (default zoom is 2.0)
    python extract_and_interpret_images.py doc.pdf -o ./visual_claims --zoom 3.0
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
import anthropic
from rapidfuzz import fuzz

from revisto_evidence_aligned.utils.secrets import get_anthropic_api_key


# Match threshold for Levenshtein similarity (0-1)
# Lower threshold (0.7) to catch matches with minor wording variations
MATCH_THRESHOLD = 0.7

# Image categories that should be interpreted with Opus
DATA_VISUALIZATION_TYPES = {"chart", "table", "graph", "infographic", "diagram"}

# Classification model (fast, cheap)
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"

# Interpretation model (detailed, accurate)
INTERPRETATION_MODEL = "claude-opus-4-20250514"


def classify_image_with_claude(
    image_bytes: bytes,
    image_ext: str,
    model: str = CLASSIFICATION_MODEL
) -> dict:
    """
    Classify an image using Claude Haiku to determine if it contains data visualizations.

    Args:
        image_bytes: Raw image bytes
        image_ext: Image extension (png, jpg, etc.)
        model: Claude model to use (default: Haiku for speed)

    Returns:
        Dict with 'category' and 'description' keys
    """
    client = anthropic.Anthropic()

    # Encode image to base64
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Map extension to media type
    media_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    media_type = media_type_map.get(image_ext.lower(), "image/png")

    prompt = """Classify this image into ONE of these categories:
- chart: Bar chart, line chart, pie chart, histogram, or similar data visualization
- table: Data table with rows and columns
- graph: Scientific plot, scatter plot, forest plot, Kaplan-Meier curve
- infographic: Visual representation combining data, text, and graphics
- diagram: Flow chart, process diagram, anatomical diagram, mechanism of action
- logo: Company logo, brand mark, or icon
- photograph: Photo of people, places, or objects
- decorative: Background image, pattern, or decorative element
- text: Image containing primarily text (screenshot of text, slide)
- other: Anything that doesn't fit above

Respond with ONLY a JSON object:
{"category": "<category>", "description": "<brief 5-10 word description>"}"""

    message = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    response_text = message.content[0].text.strip()

    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"category": "other", "description": response_text[:50]}
    except json.JSONDecodeError:
        return {"category": "other", "description": response_text[:50]}


def extract_bboxes_from_pdf(pdf_path: str) -> dict:
    """
    Extract bounding boxes for all images in a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict mapping (page_num, image_index) to bbox tuple
    """
    doc = fitz.open(pdf_path)
    bboxes = {}

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        valid_img_index = 0

        for img_info in image_list:
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                width = base_image["width"]
                height = base_image["height"]

                # Skip very small images (same logic as extract_images_from_pdf)
                if width < 50 or height < 50:
                    continue

                valid_img_index += 1
                img_rects = page.get_image_rects(xref)

                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    bboxes[(page_num + 1, valid_img_index)] = bbox

            except Exception:
                continue

    doc.close()
    return bboxes


def merge_bboxes_with_results(results: List[Dict], bboxes: dict) -> List[Dict]:
    """
    Merge bbox data into results that don't have it.

    Args:
        results: List of interpretation results
        bboxes: Dict mapping (page_num, image_index) to bbox

    Returns:
        Results with bbox added
    """
    for result in results:
        if result.get("bbox") is None and result.get("source") == "embedded":
            key = (result["page_num"], result.get("image_index"))
            if key in bboxes:
                result["bbox"] = bboxes[key]
    return results


def extract_images_from_pdf(pdf_path: str, render_from_page: bool = True, zoom: float = 2.0) -> list[dict]:
    """
    Extract all images from a PDF file.

    Args:
        pdf_path: Path to PDF file
        render_from_page: If True, render images from page (preserves transparency/colors).
                         If False, extract raw embedded image data.
        zoom: Zoom factor when rendering from page (higher = better quality)

    Returns:
        List of dicts with keys: page_num, image_index, image_bytes, width, height, ext
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                # Get image info for dimensions
                base_image = doc.extract_image(xref)
                width = base_image["width"]
                height = base_image["height"]

                # Skip very small images (likely icons or artifacts)
                if width < 50 or height < 50:
                    continue

                if render_from_page:
                    # Find the image's bounding box on the page
                    # This renders the image with proper transparency and compositing
                    img_rects = page.get_image_rects(xref)

                    if img_rects:
                        # Use the first rectangle (main placement)
                        rect = img_rects[0]

                        # Apply zoom for better quality
                        mat = fitz.Matrix(zoom, zoom)

                        # Render just the image region from the page
                        # This applies all transformations, masks, and compositing
                        clip = rect
                        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                        image_bytes = pix.tobytes("png")
                        image_ext = "png"

                        # Update dimensions to rendered size
                        width = pix.width
                        height = pix.height
                    else:
                        # Fallback to raw extraction if no rect found
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                else:
                    # Raw extraction (may have transparency/color issues)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                # Store the bounding box for annotation
                bbox = None
                if render_from_page and img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

                images.append({
                    "page_num": page_num + 1,
                    "image_index": img_index + 1,
                    "image_bytes": image_bytes,
                    "width": width,
                    "height": height,
                    "ext": image_ext,
                    "xref": xref,
                    "bbox": bbox
                })
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                continue

    doc.close()
    return images


def extract_page_as_image(pdf_path: str, page_num: int, zoom: float = 2.0) -> bytes:
    """
    Render a PDF page as an image.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        zoom: Zoom factor for rendering quality

    Returns:
        PNG image bytes
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image_bytes = pix.tobytes("png")

    doc.close()
    return image_bytes


DEFAULT_INTERPRETATION_PROMPT = """Convert this chart into declarative factual statements.

Write each fact as a complete sentence stating what the data shows. Use consistent phrasing:
- For percentages: "X% of patients on [DRUG] achieved [outcome]."
- For changes: "[DRUG] resulted in a [metric] of [value]."
- For baselines: "The mean baseline [metric] was [value]."

Include sample sizes when shown. Do not add interpretive comparisons."""


def interpret_image_with_claude(
    image_bytes: bytes,
    image_ext: str,
    prompt: str = DEFAULT_INTERPRETATION_PROMPT,
    model: str = "claude-opus-4-20250514"
) -> str:
    """
    Send an image to Claude Opus for interpretation.

    Args:
        image_bytes: Raw image bytes
        image_ext: Image extension (png, jpg, etc.)
        prompt: Prompt for Claude
        model: Claude model to use

    Returns:
        Claude's interpretation of the image
    """
    client = anthropic.Anthropic()

    # Encode image to base64
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Map extension to media type
    media_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    media_type = media_type_map.get(image_ext.lower(), "image/png")

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return message.content[0].text


def load_visual_claims(visual_claims_dir: str) -> List[Dict]:
    """
    Load visual claim images from a directory and interpret them.

    Args:
        visual_claims_dir: Directory containing visual claim images

    Returns:
        List of dicts with claim_name, image_bytes, ext, and interpretation
    """
    visual_claims_dir = Path(visual_claims_dir)
    if not visual_claims_dir.exists():
        raise FileNotFoundError(f"Visual claims directory not found: {visual_claims_dir}")

    claims = []
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

    for img_path in sorted(visual_claims_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            claims.append({
                "claim_name": img_path.stem,
                "image_path": str(img_path),
                "image_bytes": image_bytes,
                "ext": img_path.suffix[1:].lower()
            })

    return claims


def interpret_visual_claims(
    claims: List[Dict],
    model: str = INTERPRETATION_MODEL,
    classification_model: str = CLASSIFICATION_MODEL,
    prompt: str = DEFAULT_INTERPRETATION_PROMPT
) -> List[Dict]:
    """
    Classify and interpret visual claims using the same two-pass logic.

    Args:
        claims: List of visual claim dicts
        model: Interpretation model
        classification_model: Classification model
        prompt: Interpretation prompt

    Returns:
        Claims with classification and interpretation added
    """
    for i, claim in enumerate(claims):
        print(f"\nProcessing visual claim {i + 1}/{len(claims)}: {claim['claim_name']}...")

        # Classify
        try:
            print(f"  Classifying with {classification_model}...")
            classification = classify_image_with_claude(
                claim["image_bytes"],
                claim["ext"],
                model=classification_model
            )
            claim["classification"] = classification
            category = classification.get("category", "other")
            print(f"  Classification: {category}")
        except Exception as e:
            print(f"  Classification error: {e}")
            claim["classification"] = {"category": "unknown", "description": str(e)}
            category = "unknown"

        # Interpret (always interpret visual claims since they're expected to be meaningful)
        try:
            print(f"  Interpreting with {model}...")
            interpretation = interpret_image_with_claude(
                claim["image_bytes"],
                claim["ext"],
                prompt,
                model=model
            )
            claim["interpretation"] = interpretation
            print(f"  Interpretation received ({len(interpretation)} chars)")
        except Exception as e:
            print(f"  Interpretation error: {e}")
            claim["interpretation"] = f"Error: {str(e)}"

    return claims


def compare_interpretations(
    interpretation1: str,
    interpretation2: str,
    threshold: float = MATCH_THRESHOLD
) -> Dict:
    """
    Compare two interpretations using Levenshtein similarity.

    Args:
        interpretation1: First interpretation text
        interpretation2: Second interpretation text
        threshold: Minimum similarity threshold (0-1)

    Returns:
        Dict with match (bool), confidence (float), and reasoning (str)
    """
    # Normalize texts for comparison
    text1 = interpretation1.lower().strip()
    text2 = interpretation2.lower().strip()

    # Calculate similarity using token_set_ratio (handles word order differences)
    similarity = fuzz.token_set_ratio(text1, text2) / 100.0

    is_match = similarity >= threshold

    return {
        "match": is_match,
        "confidence": similarity,
        "reasoning": f"Levenshtein token_set_ratio: {similarity:.2%}"
    }


def find_visual_claim_matches(
    pdf_results: List[Dict],
    visual_claims: List[Dict],
    threshold: float = MATCH_THRESHOLD
) -> List[Dict]:
    """
    Find matches between extracted PDF images and visual claims.

    Args:
        pdf_results: Results from process_pdf with interpretations
        visual_claims: Visual claims with interpretations
        threshold: Minimum confidence threshold for a match

    Returns:
        List of match dicts with pdf_image, visual_claim, and comparison details
    """
    matches = []

    # Only compare images that have interpretations
    interpreted_results = [r for r in pdf_results if r.get("interpretation") and not r["interpretation"].startswith("Error")]
    interpreted_claims = [c for c in visual_claims if c.get("interpretation") and not c["interpretation"].startswith("Error")]

    print(f"\nComparing {len(interpreted_results)} PDF images with {len(interpreted_claims)} visual claims...")

    for pdf_img in interpreted_results:
        for claim in interpreted_claims:
            print(f"  Comparing page {pdf_img['page_num']} img {pdf_img.get('image_index', 'N/A')} with '{claim['claim_name']}'...")

            comparison = compare_interpretations(
                pdf_img["interpretation"],
                claim["interpretation"],
                threshold=threshold
            )

            if comparison.get("match") and comparison.get("confidence", 0) >= threshold:
                print(f"    MATCH! Confidence: {comparison['confidence']:.2f}")
                matches.append({
                    "pdf_image": {
                        "page_num": pdf_img["page_num"],
                        "image_index": pdf_img.get("image_index"),
                        "bbox": pdf_img.get("bbox"),
                        "interpretation_preview": pdf_img["interpretation"][:200]
                    },
                    "visual_claim": {
                        "name": claim["claim_name"],
                        "interpretation_preview": claim["interpretation"][:200]
                    },
                    "comparison": comparison
                })
            else:
                conf = comparison.get("confidence", 0)
                print(f"    No match (confidence: {conf:.2f})")

    return matches


def annotate_pdf_with_matches(
    pdf_path: str,
    matches: List[Dict],
    output_path: str
) -> str:
    """
    Draw rectangles on PDF pages where visual claims matched and annotate with claim names.

    Args:
        pdf_path: Path to the original PDF
        matches: List of matches from find_visual_claim_matches
        output_path: Path to save the annotated PDF

    Returns:
        Path to the annotated PDF
    """
    doc = fitz.open(pdf_path)

    # Group matches by page
    matches_by_page = {}
    for match in matches:
        page_num = match["pdf_image"]["page_num"]
        if page_num not in matches_by_page:
            matches_by_page[page_num] = []
        matches_by_page[page_num].append(match)

    for page_num, page_matches in matches_by_page.items():
        page = doc[page_num - 1]  # Convert to 0-indexed

        for match in page_matches:
            bbox = match["pdf_image"].get("bbox")
            claim_name = match["visual_claim"]["name"]
            confidence = match["comparison"].get("confidence", 0)

            if bbox:
                rect = fitz.Rect(bbox)

                # Draw rectangle around the matched image
                page.draw_rect(
                    rect,
                    color=(1, 0, 0),  # Red border
                    width=2
                )

                # Add label above the rectangle
                label = f"{claim_name} ({confidence:.0%})"
                label_rect = fitz.Rect(
                    rect.x0,
                    rect.y0 - 20,
                    rect.x0 + len(label) * 6,
                    rect.y0
                )

                # Draw label background
                page.draw_rect(label_rect, color=(1, 1, 0), fill=(1, 1, 0))  # Yellow background

                # Insert text
                page.insert_text(
                    (rect.x0 + 2, rect.y0 - 5),
                    label,
                    fontsize=10,
                    color=(0, 0, 0)  # Black text
                )

                print(f"  Annotated page {page_num}: {claim_name}")

    # Save annotated PDF
    doc.save(output_path)
    doc.close()

    print(f"\nAnnotated PDF saved to: {output_path}")
    return output_path


def process_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    save_images: bool = False,
    extract_embedded: bool = True,
    render_pages: bool = False,
    raw_extraction: bool = False,
    model: str = INTERPRETATION_MODEL,
    classification_model: str = CLASSIFICATION_MODEL,
    skip_classification: bool = False,
    zoom: float = 2.0,
    prompt: str = DEFAULT_INTERPRETATION_PROMPT
) -> list[dict]:
    """
    Process a PDF: extract images, classify them, and interpret data visualizations with Claude.

    Two-pass approach:
    1. Classify each image with Haiku (fast) to identify charts/tables/graphs
    2. Interpret only data visualizations with Opus (detailed extraction)

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save outputs (optional)
        save_images: Whether to save extracted images to disk
        extract_embedded: Extract embedded images from PDF
        render_pages: Render each page as an image (for image-based PDFs)
        raw_extraction: Use raw image extraction instead of rendering from page
                       (raw may have transparency/color issues)
        model: Claude model to use for interpretation (default: Opus)
        classification_model: Claude model for classification (default: Haiku)
        skip_classification: If True, interpret all images without classification
        zoom: Zoom factor for rendering quality (higher = better quality)
        prompt: Prompt for Claude interpretation

    Returns:
        List of results with image info, classification, and interpretations
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if extract_embedded:
        print(f"Extracting embedded images from {pdf_path.name}...")
        images = extract_images_from_pdf(str(pdf_path), render_from_page=not raw_extraction, zoom=zoom)
        print(f"Found {len(images)} embedded images")

        for i, img in enumerate(images):
            print(f"\nProcessing image {i + 1}/{len(images)} (page {img['page_num']}, {img['width']}x{img['height']})...")

            # Save image if requested
            if save_images and output_dir:
                img_filename = f"page{img['page_num']}_img{img['image_index']}.{img['ext']}"
                img_path = output_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(img["image_bytes"])
                print(f"  Saved: {img_path}")

            # Step 1: Classify image (unless skipped)
            classification = None
            should_interpret = True

            if not skip_classification:
                try:
                    print(f"  Classifying with {classification_model}...")
                    classification = classify_image_with_claude(
                        img["image_bytes"],
                        img["ext"],
                        model=classification_model
                    )
                    category = classification.get("category", "other")
                    description = classification.get("description", "")
                    print(f"  Classification: {category} - {description}")

                    # Only interpret data visualizations
                    should_interpret = category in DATA_VISUALIZATION_TYPES
                    if not should_interpret:
                        print(f"  Skipping interpretation (not a data visualization)")
                except Exception as e:
                    print(f"  Classification error: {e}, will interpret anyway")
                    classification = {"category": "unknown", "description": f"Error: {str(e)}"}

            # Step 2: Interpret with Opus (only for data visualizations)
            interpretation = None
            if should_interpret:
                try:
                    print(f"  Interpreting with {model}...")
                    interpretation = interpret_image_with_claude(
                        img["image_bytes"],
                        img["ext"],
                        prompt,
                        model=model
                    )
                    print(f"  Interpretation received ({len(interpretation)} chars)")
                except Exception as e:
                    print(f"  Error interpreting image: {e}")
                    interpretation = f"Error: {str(e)}"

            result = {
                "source": "embedded",
                "page_num": img["page_num"],
                "image_index": img["image_index"],
                "width": img["width"],
                "height": img["height"],
                "format": img["ext"],
                "bbox": img.get("bbox"),
            }

            if classification:
                result["classification"] = classification

            if interpretation:
                result["interpretation"] = interpretation

            results.append(result)

    if render_pages:
        print(f"\nRendering pages as images...")
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()

        for page_num in range(1, num_pages + 1):
            print(f"\nRendering page {page_num}/{num_pages}...")

            try:
                page_image = extract_page_as_image(str(pdf_path), page_num, zoom=zoom)

                # Save rendered page if requested
                if save_images and output_dir:
                    img_path = output_dir / f"page{page_num}_rendered.png"
                    with open(img_path, "wb") as f:
                        f.write(page_image)
                    print(f"  Saved: {img_path}")

                # For rendered pages, always interpret (they contain document content)
                print(f"  Interpreting with {model}...")
                interpretation = interpret_image_with_claude(
                    page_image,
                    "png",
                    prompt,
                    model=model
                )
                print(f"  Interpretation received ({len(interpretation)} chars)")

                results.append({
                    "source": "rendered_page",
                    "page_num": page_num,
                    "format": "png",
                    "interpretation": interpretation
                })
            except Exception as e:
                print(f"  Error processing page {page_num}: {e}")
                results.append({
                    "source": "rendered_page",
                    "page_num": page_num,
                    "format": "png",
                    "interpretation": f"Error: {str(e)}"
                })

    # Save results to JSON
    if output_dir:
        results_path = output_dir / f"{pdf_path.stem}_interpretations.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from PDF, interpret with Claude, and optionally match against visual claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf -o ./output
      Extract images, classify with Haiku, interpret charts/tables with Opus

  %(prog)s document.pdf -o ./output --save-images
      Also save extracted images as PNG files

  %(prog)s document.pdf -o ./output --visual-claims ./visual_claims
      Match extracted images against visual claims and annotate PDF

  %(prog)s scanned.pdf -o ./output --render-pages --no-embedded
      For scanned/image-based PDFs, render full pages instead of embedded images

  %(prog)s clinical.pdf -o ./output -p "Extract all statistics and p-values"
      Use custom prompt for targeted data extraction

  %(prog)s doc.pdf -o ./output --no-classify
      Skip classification, interpret ALL images with Opus (slower, more expensive)
"""
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save JSON results and images (default: current directory)",
        default="."
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save extracted images as PNG files alongside JSON results"
    )
    parser.add_argument(
        "--render-pages",
        action="store_true",
        help="Render each page as a full image (useful for scanned/image-based PDFs)"
    )
    parser.add_argument(
        "--no-embedded",
        action="store_true",
        help="Skip extraction of embedded images (use with --render-pages for scanned PDFs)"
    )
    parser.add_argument(
        "--raw-extraction",
        action="store_true",
        help="Extract raw image data instead of rendering from page (may cause transparency/color issues)"
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip classification step and interpret ALL images with Opus (slower, more expensive)"
    )
    parser.add_argument(
        "--visual-claims",
        help="Directory containing visual claim images to match against extracted images",
        default=None
    )
    parser.add_argument(
        "--load-interpretations",
        help="Load existing PDF interpretations from JSON file instead of re-extracting",
        default=None
    )
    parser.add_argument(
        "--load-visual-claims",
        help="Load existing visual claims interpretations from JSON file instead of re-interpreting",
        default=None
    )
    parser.add_argument(
        "--match-threshold",
        help="Minimum confidence threshold for visual claim matching (default: 0.7)",
        type=float,
        default=MATCH_THRESHOLD
    )
    parser.add_argument(
        "--interpretation-model",
        help="Claude model for detailed interpretation (default: claude-opus-4-20250514)",
        default=INTERPRETATION_MODEL
    )
    parser.add_argument(
        "--classification-model",
        help="Claude model for fast classification (default: claude-haiku-4-5-20251001)",
        default=CLASSIFICATION_MODEL
    )
    parser.add_argument(
        "--zoom", "-z",
        help="Zoom factor for image rendering quality (default: 2.0, higher = better quality)",
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt for Claude interpretation",
        default=DEFAULT_INTERPRETATION_PROMPT
    )

    args = parser.parse_args()

    # Get API key from secrets or environment
    api_key = get_anthropic_api_key()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or AWS Secrets Manager")
        sys.exit(1)

    # Set it for the anthropic client
    os.environ["ANTHROPIC_API_KEY"] = api_key

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing interpretations or extract new ones
    if args.load_interpretations:
        print(f"Loading existing interpretations from: {args.load_interpretations}")
        with open(args.load_interpretations, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} image interpretations")

        # Extract bboxes from PDF and merge (needed for annotation)
        print(f"Extracting bboxes from PDF for annotation...")
        bboxes = extract_bboxes_from_pdf(args.pdf_path)
        results = merge_bboxes_with_results(results, bboxes)
        print(f"Merged {len(bboxes)} bboxes with results")

        # Summary
        total_images = len(results)
        interpreted = sum(1 for r in results if "interpretation" in r)
        print(f"\n{'='*60}")
        print(f"Loaded {total_images} images ({interpreted} with interpretations)")
        print(f"{'='*60}")
    else:
        results = process_pdf(
            pdf_path=args.pdf_path,
            output_dir=str(output_dir),
            save_images=args.save_images,
            extract_embedded=not args.no_embedded,
            render_pages=args.render_pages,
            raw_extraction=args.raw_extraction,
            model=args.interpretation_model,
            classification_model=args.classification_model,
            skip_classification=args.no_classify,
            zoom=args.zoom,
            prompt=args.prompt
        )

        # Summary statistics
        total_images = len(results)
        classified = sum(1 for r in results if "classification" in r)
        interpreted = sum(1 for r in results if "interpretation" in r)

        print(f"\n{'='*60}")
        print(f"Processed {total_images} images")
        if classified > 0:
            print(f"  - Classified: {classified}")
            print(f"  - Interpreted (data visualizations): {interpreted}")
            print(f"  - Skipped (logos, photos, etc.): {classified - interpreted}")
        print(f"{'='*60}")

        for result in results:
            category = result.get("classification", {}).get("category", "N/A")
            print(f"\n--- {result['source'].upper()}: Page {result['page_num']} [{category}] ---")
            if "interpretation" in result:
                interp = result["interpretation"]
                print(interp[:500] + "..." if len(interp) > 500 else interp)
            else:
                desc = result.get("classification", {}).get("description", "No interpretation")
                print(f"[Skipped] {desc}")

    # Visual claim matching (if enabled)
    if args.visual_claims or args.load_visual_claims:
        print(f"\n{'='*60}")
        print("VISUAL CLAIM MATCHING")
        print(f"{'='*60}")

        # Load visual claims - either from JSON or from directory
        if args.load_visual_claims:
            print(f"\nLoading existing visual claims interpretations from: {args.load_visual_claims}")
            with open(args.load_visual_claims, "r", encoding="utf-8") as f:
                visual_claims = json.load(f)
            print(f"Loaded {len(visual_claims)} visual claims with interpretations")
        else:
            # Load and interpret visual claims from directory
            print(f"\nLoading visual claims from: {args.visual_claims}")
            visual_claims = load_visual_claims(args.visual_claims)
            print(f"Found {len(visual_claims)} visual claim images")

            # Interpret visual claims
            visual_claims = interpret_visual_claims(
                visual_claims,
                model=args.interpretation_model,
                classification_model=args.classification_model,
                prompt=args.prompt
            )

            # Save visual claims interpretations
            claims_json_path = output_dir / "visual_claims_interpretations.json"
            claims_for_json = [
                {k: v for k, v in c.items() if k != "image_bytes"}
                for c in visual_claims
            ]
            with open(claims_json_path, "w", encoding="utf-8") as f:
                json.dump(claims_for_json, f, indent=2, ensure_ascii=False)
            print(f"\nVisual claims interpretations saved to: {claims_json_path}")

        # Find matches
        matches = find_visual_claim_matches(
            results,
            visual_claims,
            threshold=args.match_threshold
        )

        # Save matches
        matches_json_path = output_dir / "visual_claim_matches.json"
        with open(matches_json_path, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        print(f"Matches saved to: {matches_json_path}")

        # Annotate PDF with matches
        if matches:
            pdf_name = Path(args.pdf_path).stem
            annotated_pdf_path = output_dir / f"{pdf_name}_annotated.pdf"
            annotate_pdf_with_matches(
                args.pdf_path,
                matches,
                str(annotated_pdf_path)
            )

            print(f"\n{'='*60}")
            print(f"MATCHING SUMMARY")
            print(f"{'='*60}")
            print(f"Found {len(matches)} visual claim matches:")
            for match in matches:
                claim_name = match["visual_claim"]["name"]
                page = match["pdf_image"]["page_num"]
                conf = match["comparison"]["confidence"]
                print(f"  - '{claim_name}' matched on page {page} (confidence: {conf:.0%})")
        else:
            print("\nNo visual claim matches found.")


if __name__ == "__main__":
    main()
