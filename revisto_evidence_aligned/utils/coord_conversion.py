"""Coordinate conversion utilities for LandingAI -> PyMuPDF"""


def landingai_to_pymupdf_coordinates(box: dict, page) -> tuple:
    """
    Convert a normalized LandingAI bounding box to PyMuPDF (x0, y0, x1, y1)

    Args:
        box (dict): LandingAI box with keys 'left', 'top', 'right', 'bottom' (normalized [0.0 - 1.0])
        page (fitz.Page): PyMuPDF page object

    Returns:
        tuple: (x0, y0, x1, y1) in PDF points (absolute coordinates)
    """
    page_width = page.rect.width
    page_height = page.rect.height

    # Normalize to absolute
    abs_left = box["left"] * page_width
    abs_right = box["right"] * page_width
    abs_top = box["top"] * page_height
    abs_bottom = box["bottom"] * page_height

    # Convert top-down Y to bottom-up Y (flip Y)
    y0 = page_height - abs_bottom  # bottom
    y1 = page_height - abs_top     # top

    x0 = abs_left
    x1 = abs_right

    return (x0, y0, x1, y1)
