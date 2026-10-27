"""Keep extraction, raster crops and edits in displayed-page coordinates."""
from contextlib import contextmanager

import pymupdf as fitz


def configure_text_geometry():
    """Use the same glyph metrics for source ownership, fitting and QA.

    PyMuPDF 2.0 auto-imports pymupdf4llm, which disables quad corrections as
    an import side effect. That also bypasses small-glyph-height corrections,
    particularly enlarging Noto CJK boxes beyond their allocated text frame.
    Restore our explicit policy at operation boundaries, including standalone
    helpers, rather than depending on which optional module was imported first.
    """
    fitz.TOOLS.unset_quad_corrections(False)
    fitz.TOOLS.set_small_glyph_heights(True)


def normalize_page_rotation(page):
    if not page.rotation:
        return
    # PyMuPDF 1.28.2 remove_rotation uses the MediaBox dimensions, but its
    # derotation matrix uses the CropBox. Make those agree temporarily, then
    # restore the transformed visible crop. Otherwise cropped rotated slides
    # expand to the MediaBox or their content shifts inside the crop.
    crop, media = page.cropbox, page.mediabox
    page.set_cropbox(fitz.Rect(media.x0, 0, media.x1, media.height))
    visible = (crop - fitz.Rect(media.x0, 0, media.x0, 0)) * page.rotation_matrix
    page.remove_rotation()
    full = page.cropbox
    page.set_cropbox(visible + fitz.Rect(full.x0, full.y0, full.x0, full.y0))


def normalize_page_rotations(pdf):
    """Bake /Rotate into page content in memory, preserving vectors and pixels.

    PyMuPDF text/drawing boxes are unrotated but pixmap clips and Docling boxes
    follow the displayed page. Removing the rotation flag alone would rotate
    the artwork; remove_rotation transforms the content and page boxes too.
    Never save this handle over the source or use it for bilingual originals.
    """
    for page in pdf:
        normalize_page_rotation(page)


@contextmanager
def display_pdf(source):
    configure_text_geometry()
    with fitz.open(source) as pdf:
        normalize_page_rotations(pdf)
        yield pdf
