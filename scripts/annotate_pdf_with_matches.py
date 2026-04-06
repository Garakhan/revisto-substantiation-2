#!/usr/bin/env python3
"""
Annotate PDF with visual claim matches.
Draws rectangles around matched images and adds labels with visual claim names.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict


def load_matches(matches_path: str) -> list:
    """Load visual claim matches from JSON file."""
    with open(matches_path, 'r') as f:
        return json.load(f)


def group_matches_by_pdf_location(matches: list) -> dict:
    """
    Group matches by PDF location (page_num, image_index).
    Returns dict with (page_num, bbox tuple) as key and list of visual claim names as value.
    """
    location_matches = defaultdict(set)

    for match in matches:
        pdf_image = match['pdf_image']
        visual_claim = match['visual_claim']
        confidence = match['comparison']['confidence']

        page_num = pdf_image['page_num']
        bbox = tuple(pdf_image['bbox'])
        claim_name = visual_claim['name']

        # Store claim name with confidence
        location_matches[(page_num, bbox)].add((claim_name, confidence))

    return location_matches


def get_best_match_per_location(location_matches: dict) -> dict:
    """Get the best (highest confidence) match for each location."""
    best_matches = {}
    for (page_num, bbox), claims in location_matches.items():
        # Sort by confidence (descending) and take the best
        sorted_claims = sorted(claims, key=lambda x: x[1], reverse=True)
        best_match = sorted_claims[0]
        best_matches[(page_num, bbox)] = {
            'claim_name': best_match[0],
            'confidence': best_match[1],
            'all_claims': [c[0] for c in sorted_claims]
        }
    return best_matches


def annotate_pdf(pdf_path: str, output_path: str, best_matches: dict):
    """
    Annotate PDF with rectangles and labels for each matched location.
    """
    doc = fitz.open(pdf_path)

    for (page_num, bbox), match_info in best_matches.items():
        # PyMuPDF uses 0-based page indexing
        page_idx = page_num - 1

        if page_idx < 0 or page_idx >= len(doc):
            print(f"Warning: Page {page_num} out of range, skipping")
            continue

        page = doc[page_idx]

        # The bbox from the JSON is [x0, y0, x1, y1]
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)

        # Draw rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=(1, 0, 0))  # Red border
        annot.set_border(width=2)
        annot.update()

        # Add text annotation with claim name
        claim_name = match_info['claim_name']
        confidence = match_info['confidence']

        # Create label text
        label = f"{claim_name} ({confidence:.0%})"

        # Add text annotation above the rectangle
        text_point = fitz.Point(x0, y0 - 5)

        # Insert text as a free text annotation
        text_rect = fitz.Rect(x0, y0 - 18, x0 + len(label) * 5, y0 - 2)
        text_annot = page.add_freetext_annot(
            text_rect,
            label,
            fontsize=8,
            fontname="helv",
            text_color=(1, 0, 0),  # Red text
            fill_color=(1, 1, 1),  # White background
        )
        text_annot.update()

        print(f"Annotated page {page_num}: {claim_name} (confidence: {confidence:.2%})")

    # Save annotated PDF
    doc.save(output_path)
    doc.close()
    print(f"\nAnnotated PDF saved to: {output_path}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    matches_path = script_dir / "outputs" / "visual_claim_matches.json"
    pdf_path = script_dir / "pp-tr-us-2479.pdf"
    output_path = script_dir / "outputs" / "pp-tr-us-2479_visual_claims_annotated.pdf"

    print(f"Loading matches from: {matches_path}")
    matches = load_matches(matches_path)
    print(f"Found {len(matches)} total matches")

    # Group matches by PDF location
    location_matches = group_matches_by_pdf_location(matches)
    print(f"Found {len(location_matches)} unique PDF locations with matches")

    # Get best match per location
    best_matches = get_best_match_per_location(location_matches)

    print(f"\nAnnotating PDF: {pdf_path}")
    annotate_pdf(str(pdf_path), str(output_path), best_matches)

    # Print summary
    print("\n=== Annotation Summary ===")
    for (page_num, bbox), match_info in sorted(best_matches.items()):
        print(f"Page {page_num}: {match_info['claim_name']} ({match_info['confidence']:.2%})")
        if len(match_info['all_claims']) > 1:
            print(f"  Also matched: {', '.join(match_info['all_claims'][1:])}")


if __name__ == "__main__":
    main()
