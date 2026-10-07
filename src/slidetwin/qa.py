from __future__ import annotations

from pathlib import Path
from collections import Counter
from copy import deepcopy
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz

from .models import digest, write_json
from .render import Placement


QA_KINDS = {
    "page_count", "page_dimensions", "raster_dimensions", "original_page_pixels_changed",
    "graphics_outside_edit_regions_changed", "inserted_text_not_extractable",
}


def page_indices(selected: list[int], bilingual: bool = True) -> dict[int, dict]:
    """Physical PDF page numbers, independent of printed slide footers."""
    return {number: {"source_page": number, "chinese_page": index + 1,
                     "bilingual_original_page": index * 2 + 1,
                     "bilingual_translated_page": index * 2 + 2,
                     "translated_page": index * 2 + 2 if bilingual else index + 1}
            for index, number in enumerate(selected)}


def geometric_text(page: fitz.Page, frame: fitz.Rect) -> str:
    """Read only the target's characters, joining spans on the same baseline.

    Content-stream order and emphasis spans do not define reading order. Use
    glyph geometry; never accept an unordered bag of characters as a sentence.
    """
    rows = []
    for block in page.get_text("rawdict", clip=frame)["blocks"]:
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span["chars"]
                     if frame.contains(fitz.Point((c["bbox"][0] + c["bbox"][2]) / 2,
                                                  (c["bbox"][1] + c["bbox"][3]) / 2))]
            if chars:
                direction = tuple(round(v, 3) for v in line.get("dir", (1, 0)))
                baseline = chars[0]["origin"][1] * direction[0] - chars[0]["origin"][0] * direction[1]
                height = min(fitz.Rect(c["bbox"]).height for c in chars)
                row = next((r for r in rows if r['direction'] == direction and
                            abs(r['baseline'] - baseline) <= max(.5, min(r['height'], height) * .15)), None)
                if row is None:
                    rows.append({'direction': direction, 'baseline': baseline, 'height': height, 'chars': chars})
                else:
                    row['chars'].extend(chars)
    lines = []
    for row in rows:
        direction = row['direction']
        chars = sorted(row['chars'], key=lambda c: c["origin"][0] * direction[0] + c["origin"][1] * direction[1])
        lines.append((min(c["bbox"][1] for c in chars), min(c["bbox"][0] for c in chars),
                      "".join(c["c"] for c in chars)))
    return "\n".join(text for _, _, text in sorted(lines))


def text_layer_check(page: fitz.Page, entry: Placement) -> dict | None:
    # An asset-only formula intentionally has no extractable characters. Its
    # visual preservation is checked separately from the text layer.
    expected = compact(entry.extractable_text if entry.assets else entry.extractable_text or entry.text)
    if not expected:
        return None
    frame = fitz.Rect(entry.frame) + (-1.8, -1.8, 1.8, 1.8)
    observed = compact(page.get_text(clip=frame))
    if expected in observed or expected in compact(geometric_text(page, frame)):
        return None
    unmapped = sum(c[0] in {0, 0xFFFD} for span in page.get_texttrace()
                   for c in span["chars"] if frame.contains(fitz.Point(c[2])))
    missing = Counter(expected) - Counter(observed)
    return {"page": entry.page, "id": entry.id, "region_ids": [entry.id],
            "kind": "inserted_text_not_extractable", "category": "text_layer",
            "reason_code": "invalid_unicode_mapping" if unmapped else "target_characters_not_recovered",
            "unmapped_glyphs": unmapped, "missing_characters": sorted(missing),
            "missing_character_count": sum(missing.values()),
            "expected_character_count": len(expected), "observed_character_count": len(observed),
            "check": "region_geometry_and_character_sequence"}


def merge_final_checks(report: dict, final_qa: dict) -> dict:
    """Replace intermediate QA conclusions with checks of the published PDF.

    Keep all historical evidence, explicitly labelled as intermediate. Layout
    and extraction issues are never discarded by a later QA pass.
    """
    report = deepcopy(report)
    previous = report.get("failures", [])
    historical = [dict(i, phase="intermediate") for i in previous if i.get("kind") in QA_KINDS]
    if historical:
        report.setdefault("intermediate_diagnostics", []).extend(historical)
    report["failures"] = [i for i in previous if i.get("kind") not in QA_KINDS] + deepcopy(final_qa.get("failures", []))
    pages = {p["source_page"]: p for p in report.get("pages", []) if "source_page" in p}
    for checked in final_qa.get("pages", []):
        number = checked["source_page"]
        page = pages.setdefault(number, {"source_page": number, "issues": []})
        old = page.get("issues", [])
        intermediate = [dict(i, phase="intermediate") for i in old if i.get("kind") in QA_KINDS]
        if intermediate:
            page.setdefault("intermediate_diagnostics", []).extend(intermediate)
        page["issues"] = [i for i in old if i.get("kind") not in QA_KINDS]
        page.update({key: checked[key] for key in ("chinese_page", "bilingual_original_page",
                    "bilingual_translated_page", "translated_page") if key in checked})
    for failure in final_qa.get("failures", []):
        number = failure.get("source_page", failure.get("page"))
        if isinstance(number, int) and number not in pages:
            pages[number] = {key: failure[key] for key in ("source_page", "chinese_page",
                             "bilingual_original_page", "bilingual_translated_page", "translated_page")
                             if key in failure}
            pages[number]["source_page"] = number
        if number in pages:
            pages[number].setdefault("issues", []).append(dict(failure, phase="final"))
    report["pages"] = list(pages.values())
    report["final_qa"] = deepcopy(final_qa)
    return report


def compact(text: str) -> str:
    # Story's font shaping uses Unicode presentation forms for ligatures and
    # East Asian punctuation. Expand only these known typographical forms.
    # Broad NFKC would silently equate mathematical letters and superscripts.
    for shaped, characters in {'\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi',
                               '\ufb04': 'ffl', '\ufb05': 'st', '\ufb06': 'st',
                               '\u2e3a': '\u2014\u2014', '\u2e3b': '\u2014\u2014\u2014',
                               '\u30fb': '\u00b7'}.items():
        text = text.replace(shaped, characters)
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    return re.sub(r"\s+|[\u200b\u200c\u200d\ufeff]", "", text)


def edge_rounding_pixel_count(left: np.ndarray, right: np.ndarray) -> int | None:
    """Count harmless one-level outer-edge rounding; reject every other change."""
    if left.shape != right.shape:return None
    delta=np.max(np.abs(left.astype(np.int16)-right.astype(np.int16)),axis=2)
    if delta.max()<=1 and not np.any(delta[1:-1,1:-1]):
        return int(np.count_nonzero(delta))
    return None


def verify(source: Path, output: Path, selected: list[int], placements: list[Placement], work: Path,
           bilingual=True, *, phase="final") -> dict:
    failures, page_reports = [], []
    indices = page_indices(selected, bilingual)
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
                if (op.width, op.height, op.n) != (tp.width, tp.height, tp.n):
                    failures.append({"page": number, "kind": "raster_dimensions"})
                    continue
                a = np.frombuffer(op.samples, np.uint8).reshape(op.height, op.width, op.n).astype(np.int16)
                b = np.frombuffer(tp.samples, np.uint8).reshape(tp.height, tp.width, tp.n).astype(np.int16)
                mask = np.zeros((op.height, op.width), bool)
                page_entries = [x for x in placements if x.page == number]
                for entry in page_entries:
                    # Account for antialiasing only, never mask the entire slide.
                    for box in [entry.frame, entry.source_bbox, *entry.erase, *(entry.shadow_boxes or []), *([entry.raster_patch_box] if entry.raster_patch_box else []), *[d['bbox'] for d in entry.decorations or []]]:
                        r = fitz.Rect(box) + (-1.8, -1.8, 1.8, 1.8)
                        x0, y0 = max(0, int(r.x0*96/72)), max(0, int(r.y0*96/72))
                        x1, y1 = min(op.width, int(r.x1*96/72)+1), min(op.height, int(r.y1*96/72)+1)
                        mask[y0:y1, x0:x1] = True
                outside = np.any(np.abs(a-b) > 20, axis=2) & ~mask
                changed = int(outside.sum())
                if changed > 6:
                    failures.append({"page": number, "kind": "graphics_outside_edit_regions_changed", "pixels": changed})
                for entry in page_entries:
                    if failure := text_layer_check(target, entry):
                        failures.append(failure)
                page_reports.append({**indices[number],
                                     "outside_edit_changed_pixels": changed, "placements": len(page_entries),
                                     "original_edge_rounding_pixels_max_delta_1":original_edge_rounding})
    for failure in failures:
        failure.update(indices.get(failure.get("page"), {}), phase=phase)
        failure.setdefault("region_ids", [failure["id"]] if failure.get("id") else [])
        failure.setdefault("category", "document_structure" if failure["kind"] in {
            "page_count", "page_dimensions", "raster_dimensions"} else "graphics")
    report = {"passed": not failures, "source_sha256": digest(source.read_bytes()), "output_sha256": digest(output.read_bytes()),
              "phase": phase, "output_mode": "bilingual" if bilingual else "chinese",
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
