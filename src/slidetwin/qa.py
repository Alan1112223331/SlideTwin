from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz

from .models import digest, write_json
from .render import Placement


def compact(text: str) -> str:
    # MuPDF Story's shaping maps ASCII hyphens to a typographic hyphen in
    # its text layer. Normalize only that equivalent, not digits or formulas.
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    return re.sub(r"\s+|[\u200b\u200c\u200d\ufeff]", "", text)


def edge_rounding_pixel_count(left: np.ndarray, right: np.ndarray) -> int | None:
    """Count harmless one-level outer-edge rounding; reject every other change."""
    if left.shape != right.shape:return None
    delta=np.max(np.abs(left.astype(np.int16)-right.astype(np.int16)),axis=2)
    if delta.max()<=1 and not np.any(delta[1:-1,1:-1]):
        return int(np.count_nonzero(delta))
    return None


def verify(source: Path, output: Path, selected: list[int], placements: list[Placement], work: Path, bilingual=True) -> dict:
    failures, page_reports = [], []
    with fitz.open(source) as src, fitz.open(output) as result:
        expected = len(selected) * (2 if bilingual else 1)
        if len(result) != expected:
            failures.append({"kind": "page_count", "expected": expected, "actual": len(result)})
        else:
            for index, number in enumerate(selected):
                original = src[number-1]
                target = result[index*2+1 if bilingual else index]
                if original.rect != target.rect:
                    failures.append({"page": number, "kind": "page_dimensions"})
                    continue
                op = original.get_pixmap(dpi=96, alpha=False)
                original_edge_rounding = 0
                if bilingual:
                    odd = result[index*2].get_pixmap(dpi=96, alpha=False)
                    if op.samples != odd.samples:
                        left=np.frombuffer(op.samples,np.uint8).reshape(op.height,op.width,op.n).astype(np.int16)
                        right=np.frombuffer(odd.samples,np.uint8).reshape(odd.height,odd.width,odd.n).astype(np.int16)
                        # PDF serialization can round a full-bleed rectangle at
                        # the page edge by one color level. Never allow an
                        # interior difference or any larger edge difference.
                        rounding=edge_rounding_pixel_count(left,right)
                        if rounding is not None:
                            original_edge_rounding=rounding
                        else:
                            failures.append({"page": number, "kind": "original_page_pixels_changed"})
                tp = target.get_pixmap(dpi=96, alpha=False)
                a = np.frombuffer(op.samples, np.uint8).reshape(op.height, op.width, op.n).astype(np.int16)
                b = np.frombuffer(tp.samples, np.uint8).reshape(tp.height, tp.width, tp.n).astype(np.int16)
                mask = np.zeros((op.height, op.width), bool)
                page_entries = [x for x in placements if x.page == number]
                for entry in page_entries:
                    # Account for antialiasing only, never mask the entire slide.
                    for box in [entry.frame, entry.source_bbox, *entry.erase, *(entry.shadow_boxes or []), *([entry.raster_patch_box] if entry.raster_patch_box else [])]:
                        r = fitz.Rect(box) + (-1.8, -1.8, 1.8, 1.8)
                        x0, y0 = max(0, int(r.x0*96/72)), max(0, int(r.y0*96/72))
                        x1, y1 = min(op.width, int(r.x1*96/72)+1), min(op.height, int(r.y1*96/72)+1)
                        mask[y0:y1, x0:x1] = True
                outside = np.any(np.abs(a-b) > 20, axis=2) & ~mask
                changed = int(outside.sum())
                if changed > 6:
                    failures.append({"page": number, "kind": "graphics_outside_edit_regions_changed", "pixels": changed})
                output_text = compact(target.get_text())
                for entry in page_entries:
                    if compact(entry.extractable_text or entry.text) not in output_text:
                        failures.append({"page": number, "id": entry.id, "kind": "inserted_text_not_extractable"})
                page_reports.append({"source_page": number, "translated_page": index*2+2 if bilingual else index+1,
                                     "outside_edit_changed_pixels": changed, "placements": len(page_entries),
                                     "original_edge_rounding_pixels_max_delta_1":original_edge_rounding})
    report = {"passed": not failures, "source_sha256": digest(source.read_bytes()), "output_sha256": digest(output.read_bytes()),
              "pages": page_reports, "failures": failures, "visual_review": "pending_human_review"}
    write_json(work/"qa.json", report)
    return report


def render_previews(output: Path, work: Path, selected: list[int], dpi=110, bilingual=True) -> dict:
    folder = work / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    # Poppler changes zero-padding with the page count. Remove only our own
    # numbered preview files so a later smaller/larger selection cannot mix old
    # images with this run's evidence. No source files or recursive deletes.
    folder.resolve().relative_to(work.resolve())
    for path in folder.iterdir():
        if path.is_file() and re.fullmatch(r"(?:page|pair)-\d+\.png", path.name):
            path.unlink()
    poppler = shutil.which("pdftoppm")
    if poppler:
        subprocess.run([poppler, "-r", str(dpi), "-png", str(output), str(folder/"page")], check=True, capture_output=True)
        rendered = sorted(folder.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        engine = "poppler"
    else:
        rendered = []
        with fitz.open(output) as doc:
            for i, page in enumerate(doc):
                path = folder/f"page-{i+1:04d}.png"
                page.get_pixmap(dpi=dpi, alpha=False).save(path)
                rendered.append(path)
        engine = "pymupdf"
    expected = len(selected) * (2 if bilingual else 1)
    if len(rendered) != expected:
        raise RuntimeError(f"Preview page count mismatch: {len(rendered)} != {expected}")
    contacts = []
    if bilingual:
        for index, source_page in enumerate(selected):
            with Image.open(rendered[index*2]) as left, Image.open(rendered[index*2+1]) as right:
                canvas = Image.new("RGB", (left.width+right.width+24, max(left.height, right.height)+42), "#dddddd")
                canvas.paste(left, (0, 36))
                canvas.paste(right, (left.width+24, 36))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 10), f"Source page {source_page}", fill="black")
                draw.text((left.width+36, 10), "SlideTwin translation", fill="black")
                path = folder/f"pair-{source_page:04d}.png"
                canvas.save(path)
                contacts.append(str(path.resolve()))
    manifest = {"engine": engine, "rendered_pages": len(rendered), "pairs": contacts,
                "note": "Automatic checks do not certify semantic or visual perfection. Inspect all pairs before relying on a new document type."}
    write_json(work/"preview.json", manifest)
    return manifest
