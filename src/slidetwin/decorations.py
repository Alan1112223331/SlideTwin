"""Recover native text underlines using only glyph and vector geometry.

An underline is a text style, not a line fixed to the source page's coordinates.
Ambiguous graphics and unmapped partial styles remain intact with a diagnostic.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import statistics

import numpy as np
import pymupdf as fitz


@dataclass
class DecorationPlan:
    underline: bool = False
    erasures: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


def _box(chars):
    box = fitz.Rect(chars[0]['bbox'])
    for char in chars[1:]:
        box |= fitz.Rect(char['bbox'])
    return box


def _color(integer):
    return tuple(((integer >> shift) & 255) / 255 for shift in (16, 8, 0))


def _thin_shape(drawing):
    """A single stroke or a solid axis-aligned thin rectangle/polygon."""
    items = drawing.get('items', [])
    if drawing.get('type') == 's' and len(items) == 1 and items[0][0] == 'l':
        if drawing.get('stroke_opacity', 1) != 1 or drawing.get('dashes', '[] 0') != '[] 0':
            return None
        a, b = items[0][1:]
        width = drawing.get('width', 1) or 1
        if abs(a.y - b.y) > .1 or width > 2.5:
            return None
        return fitz.Rect(min(a.x, b.x), a.y - width / 2, max(a.x, b.x), a.y + width / 2), drawing.get('color')
    if drawing.get('type') != 'f' or drawing.get('fill_opacity', 1) != 1:
        return None
    box = fitz.Rect(drawing['rect'])
    if not .1 <= box.height <= 2.5 or box.width < 8:
        return None
    if len(items) == 1 and items[0][0] == 're':
        return box, drawing.get('fill')
    if not items or any(item[0] != 'l' for item in items):
        return None
    for _, a, b in items:
        if abs(a.x - b.x) > .1 and abs(a.y - b.y) > .1:
            return None
        if not all(abs(point.y - box.y0) <= .1 or abs(point.y - box.y1) <= .1
                   or abs(point.x - box.x0) <= .1 or abs(point.x - box.x1) <= .1 for point in (a, b)):
            return None
    return box, drawing.get('fill')


def _uniform_background(page, box, size, drawing, drawings):
    """Require a uniform sampled background and no intersecting graphics.

    Samples sit below the font's descender zone, rather than mistaking a glyph's
    colored pixels for background. Never assume that a page background is white.
    """
    erase = box + (-.35, -.35, .35, .35)
    if not page.rect.contains(erase):
        return None
    if any(fitz.Rect(image['bbox']).intersects(erase) for image in page.get_image_info()):
        return None
    for other in drawings:
        bounds = fitz.Rect(other['rect'])
        stroke = (other.get('width') or 0) / 2
        if stroke:
            bounds += (-stroke, -stroke, stroke, stroke)
        if other is drawing or not bounds.intersects(erase):
            continue
        # A large opaque filled rectangle can be the measured local backdrop.
        if (other.get('type') == 'f' and other.get('fill_opacity', 1) == 1
                and len(other.get('items', [])) == 1 and other['items'][0][0] == 're'
                and fitz.Rect(other['rect']).contains(erase + (-2, -size * .5, 2, size * .5))):
            continue
        return None
    strips = [fitz.Rect(box.x0, box.y1 + max(2, size * .12), box.x1, box.y1 + max(3, size * .2)),
              fitz.Rect(box.x0 - 2.5, box.y0 - .2, box.x0 - .7, box.y1 + .2),
              fitz.Rect(box.x1 + .7, box.y0 - .2, box.x1 + 2.5, box.y1 + .2)]
    colors = []
    for strip in strips:
        strip &= page.rect
        if strip.is_empty:
            return None
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=strip, alpha=False)
        pixels = np.frombuffer(pix.samples, np.uint8).reshape(-1, pix.n)[:, :3].astype(np.int16)
        if not pixels.size:
            return None
        median = np.median(pixels, axis=0)
        if np.mean(np.max(np.abs(pixels - median), axis=1) <= 3) < .97:
            return None
        colors.append(median)
    if np.max(np.ptp(np.array(colors), axis=0)) > 3:
        return None
    return list(np.median(colors, axis=0) / 255)


def source_decorations(page, region, lines, *, create_styles=False):
    """Return unchanged-index inline styles and a :class:`DecorationPlan`.

    ``create_styles=True`` is for extraction before model calls. It appends
    native source spans for the model's existing ``_s`` style-translation
    protocol. Rendering never invents or guesses a translated partial span.
    """
    styles = deepcopy(region.inline_styles)
    plan = DecorationPlan()
    if not region.native:
        return styles, plan
    region_box = fitz.Rect(region.bbox) + (-.4, -.4, .4, .4)
    rows = []
    all_chars = []
    for line in lines:
        if any(abs(a - b) > .01 for a, b in zip(line.get('dir', [1, 0]), [1, 0])):
            continue
        chars = [c for c in line['chars'] if not c['c'].isspace()
                 and region_box.contains(fitz.Point((c['bbox'][0] + c['bbox'][2]) / 2,
                                                    (c['bbox'][1] + c['bbox'][3]) / 2))]
        if chars:
            rows.append((line, chars))
            all_chars.extend(chars)
    if not all_chars:
        return styles, plan
    drawings = page.get_drawings()
    matches = []
    for drawing in drawings:
        shape = _thin_shape(drawing)
        if shape is None:
            continue
        box, color = shape
        if color is None:
            continue
        for line, chars in rows:
            size = statistics.median(c['size'] for c in chars)
            baseline = statistics.median(c['origin'][1] for c in chars)
            if not .015 * size <= (box.y0 + box.y1) / 2 - baseline <= .18 * size:
                continue
            if box.height > .09 * size:
                continue
            covered = [c for c in chars if min(box.x1, c['bbox'][2]) - max(box.x0, c['bbox'][0])
                       >= (c['bbox'][2] - c['bbox'][0]) * .9]
            if len(covered) < 2:
                continue
            glyph_box = _box(covered)
            tolerance = max(.75, size * .075)
            if abs(box.x0 - glyph_box.x0) > tolerance or abs(box.x1 - glyph_box.x1) > tolerance:
                continue
            if any(max(abs(a - b) for a, b in zip(_color(c.get('color', region.color)), color)) > .035
                   for c in covered):
                continue
            # A rule touching another path is part of a graphic, not an isolated
            # text decoration; the background helper will reject it.
            background = _uniform_background(page, box, size, drawing, drawings)
            matches.append({'box': box, 'chars': covered, 'line': line, 'background': background})
            break
    if not matches:
        return styles, plan
    identities = {id(c) for m in matches for c in m['chars']}
    full = len(identities) >= len(all_chars) * .97
    if full and all(m['background'] is not None for m in matches):
        plan.underline = True
        plan.erasures = [{'bbox': list(m['box'] + (-.35, -.35, .35, .35)),
                         'background': m['background'], 'kind': 'source_underline'} for m in matches]
        return styles, plan
    for match in matches:
        covered = match['chars']
        box = _box(covered)
        line_chars = match['line']['chars']
        first, last = line_chars.index(covered[0]), line_chars.index(covered[-1])
        source = ''.join(c['c'] for c in line_chars[first:last + 1]).strip()
        # Restore program-protected tokens for exact character comparisons only.
        candidates = []
        for index, style in enumerate(styles):
            text = style.get('source', '')
            for token, literal in region.protected.items():
                text = text.replace(token, literal)
            if ''.join(text.split()) == ''.join(source.split()):
                candidates.append(index)
        if not candidates and create_styles:
            styles.append({'source': source, 'bbox': list(box), 'color': covered[0].get('color', region.color),
                           'bold': bool(covered[0].get('flags', 0) & 16),
                           'italic': bool(covered[0].get('flags', 0) & 2),
                           'size': statistics.median(c['size'] for c in covered), 'underline': True})
            candidates = [len(styles) - 1]
        if match['background'] is None:
            plan.warnings.append({'page': region.page, 'id': region.id, 'kind': 'underline_background_unsafe',
                                  'reason_code': 'nonuniform_background_or_intersecting_graphic'})
            continue
        if len(candidates) != 1:
            plan.warnings.append({'page': region.page, 'id': region.id, 'kind': 'underline_mapping_unavailable',
                                  'reason_code': 'partial_source_style_not_mapped'})
            continue
        styles[candidates[0]]['underline'] = True
        plan.erasures.append({'bbox': list(match['box'] + (-.35, -.35, .35, .35)),
                              'background': match['background'], 'kind': 'source_underline',
                              'style_index': candidates[0], 'source': source})
    return styles, plan
