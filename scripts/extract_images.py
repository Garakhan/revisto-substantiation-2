"""
Extract images from a PDF using PyMuPDF.

Usage:
    python extract_images.py <pdf_path> [options]

Examples:
    # Extract images and save to output directory
    python extract_images.py document.pdf -o ./extracted_images

    # Higher quality rendering (default zoom is 2.0)
    python extract_images.py document.pdf -o ./extracted_images --zoom 3.0

    # Use raw extraction instead of rendering from page
    python extract_images.py document.pdf -o ./extracted_images --raw-extraction
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import fitz  # PyMuPDF


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


def extract_images_from_pdf(pdf_path: str, render_from_page: bool = True, zoom: float = 2.0) -> List[Dict]:
    """
    Extract all images from a PDF file.

    Args:
        pdf_path: Path to PDF file
        render_from_page: If True, render images from page (preserves transparency/colors).
                         If False, extract raw embedded image data.
        zoom: Zoom factor when rendering from page (higher = better quality)

    Returns:
        List of dicts with keys: page_num, image_index, image_bytes, width, height, ext, bbox
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


def process_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    save_images: bool = True,
    extract_embedded: bool = True,
    render_pages: bool = False,
    raw_extraction: bool = False,
    zoom: float = 2.0
) -> List[Dict]:
    """
    Process a PDF and extract images.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save outputs
        save_images: Whether to save extracted images to disk
        extract_embedded: Extract embedded images from PDF
        render_pages: Render each page as an image (for image-based PDFs)
        raw_extraction: Use raw image extraction instead of rendering from page
        zoom: Zoom factor for rendering quality (higher = better quality)

    Returns:
        List of results with image info
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
            print(f"Processing image {i + 1}/{len(images)} (page {img['page_num']}, {img['width']}x{img['height']})...")

            # Save image
            if save_images and output_dir:
                img_filename = f"page{img['page_num']}_img{img['image_index']}.{img['ext']}"
                img_path = output_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(img["image_bytes"])
                print(f"  Saved: {img_path}")

            result = {
                "source": "embedded",
                "page_num": img["page_num"],
                "image_index": img["image_index"],
                "width": img["width"],
                "height": img["height"],
                "format": img["ext"],
                "bbox": img.get("bbox"),
            }

            if save_images and output_dir:
                result["file_path"] = str(output_dir / f"page{img['page_num']}_img{img['image_index']}.{img['ext']}")

            results.append(result)

    if render_pages:
        print(f"\nRendering pages as images...")
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()

        for page_num in range(1, num_pages + 1):
            print(f"Rendering page {page_num}/{num_pages}...")

            try:
                page_image = extract_page_as_image(str(pdf_path), page_num, zoom=zoom)

                # Save rendered page
                if save_images and output_dir:
                    img_path = output_dir / f"page{page_num}_rendered.png"
                    with open(img_path, "wb") as f:
                        f.write(page_image)
                    print(f"  Saved: {img_path}")

                result = {
                    "source": "rendered_page",
                    "page_num": page_num,
                    "format": "png",
                }

                if save_images and output_dir:
                    result["file_path"] = str(output_dir / f"page{page_num}_rendered.png")

                results.append(result)
            except Exception as e:
                print(f"  Error processing page {page_num}: {e}")

    # Save results to JSON
    if output_dir:
        results_path = output_dir / f"{pdf_path.stem}_extractions.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from PDF using PyMuPDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf -o ./extracted_images
      Extract embedded images and save to directory

  %(prog)s document.pdf -o ./extracted_images --zoom 3.0
      Higher quality rendering

  %(prog)s scanned.pdf -o ./extracted_images --render-pages --no-embedded
      For scanned PDFs, render full pages instead of extracting embedded images
"""
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save extracted images (default: ./extracted)",
        default="./extracted"
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
        "--zoom", "-z",
        help="Zoom factor for image rendering quality (default: 2.0, higher = better quality)",
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save images to disk, only output JSON metadata"
    )

    args = parser.parse_args()

    results = process_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        save_images=not args.no_save,
        extract_embedded=not args.no_embedded,
        render_pages=args.render_pages,
        raw_extraction=args.raw_extraction,
        zoom=args.zoom
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Extracted {len(results)} images")
    print(f"{'='*60}")

    for result in results:
        print(f"  Page {result['page_num']}: {result['source']} ({result.get('width', 'N/A')}x{result.get('height', 'N/A')})")


if __name__ == "__main__":
    main()
