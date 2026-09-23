"""Refine ambiguous Docling OCR boxes without repainting figure artwork."""
import json
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz

from .models import digest, write_json


def recognize(page, box, work):
    scale = 2.5
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box, alpha=False)
    key = digest(pix.samples + str((pix.x, pix.y)).encode() + b'precise-labels-v1')
    path = Path(work) / 'label-ocr' / f'{key}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))['rows']
    import torch
    from rapidocr import RapidOCR, EngineType
    if not hasattr(recognize, 'engine'):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        recognize.engine = RapidOCR(params={'Global.log_level': 'warning',
            'Det.engine_type': EngineType.TORCH, 'Cls.engine_type': EngineType.TORCH,
            'Rec.engine_type': EngineType.TORCH})
    result = recognize.engine(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3])
    rows = []
    for points, text, score in zip(result.boxes if result.boxes is not None else [], result.txts or [], result.scores or []):
        points = np.array(points)
        rows.append({'text': text, 'score': float(score), 'bbox': [
            (pix.x + float(points[:, 0].min())) / scale, (pix.y + float(points[:, 1].min())) / scale,
            (pix.x + float(points[:, 0].max())) / scale, (pix.y + float(points[:, 1].max())) / scale]})
    write_json(path, {'key': key, 'engine': 'RapidOCR/PyTorch', 'rows': rows})
    return rows


def caption_body_box(page, box):
    """Keep a filled blue figure-number badge and its horizontal rule intact."""
    import cv2
    scale = 3
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box + (-2, -2, 2, 2), alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.int16)
    blue = ((rgb[:, :, 2] - rgb[:, :, 0] > 35) & (rgb[:, :, 1] - rgb[:, :, 0] > 25) & (rgb[:, :, 0] < 160)).astype(np.uint8)
    original_blue=blue.copy()
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    badges = [(x, y, w, h) for x, y, w, h, area in stats[1:]
              if x < pix.width * .35 and h > 6 * scale and w > h * 1.3 and area > w * h * .45]
    if not badges:
        return None
    x, y, w, h = max(badges, key=lambda s: s[2] * s[3])
    # White badge lettering can split the blue fill after morphology. Recover
    # its full horizontal extent from the original color band; a thin rule has
    # too few blue pixels per column to count as badge background.
    counts=original_blue[y:y+h].sum(axis=0)
    columns=np.where(counts>=max(2,h*.1))[0]
    columns=columns[(columns>=x)&(columns<x+w+2*h)]
    edge=max(x+w,int(columns.max())+1) if len(columns) else x+w
    body = fitz.Rect((pix.x+edge)/scale+1.5, box.y0, box.x1+2, box.y1+2) & page.rect
    return body if body.width > 20 else None


def refine(page, descriptors, work):
    result = []
    case_pattern = re.compile(r'Case\s*\d+', re.I)
    raster_cases = [d for d in descriptors if case_pattern.fullmatch(d['text'].strip()) and not page.get_textbox(d['bbox']).strip()]
    case_rows = recognize(page, page.rect, work) if raster_cases else []
    for d in descriptors:
        text = d['text'].strip()
        if page.get_textbox(d['bbox']).strip():
            result.append(d)
            continue
        if raster_cases and case_pattern.fullmatch(text):
            continue
        # Disconnected axis strokes and formula subscripts are artwork, even if
        # an OCR detector calls them a prose object.
        if (re.fullmatch(r'(?:[cg]?[gdsb]{1,2}\s*)?\(overlap\)', text, re.I)
                or re.fullmatch(r'HH+', text) and d['bbox'].height > d['bbox'].width*2):
            continue
        if re.match(r'^FIG\b', text, re.I):
            body = caption_body_box(page, d['bbox'])
            if body is not None:
                rows = [r for r in recognize(page, body, work) if r['score'] >= .8]
                if rows:
                    # Separate OCR words on one visual baseline can have
                    # different ascender heights. Read left to right in a row.
                    ordered=[]
                    for row in sorted(rows,key=lambda r:r['bbox'][1]):
                        existing=next((g for g in ordered if abs(g[0]['bbox'][1]-row['bbox'][1])<min(g[0]['bbox'][3]-g[0]['bbox'][1],row['bbox'][3]-row['bbox'][1])*.6),None)
                        if existing is None:ordered.append([row])
                        else:existing.append(row)
                    rows=[r for group in ordered for r in sorted(group,key=lambda r:r['bbox'][0])]
                    box = fitz.Rect(rows[0]['bbox'])
                    for row in rows[1:]:
                        box |= fitz.Rect(row['bbox'])
                    result.append({**d, 'bbox': box, 'text': ' '.join(r['text'] for r in rows), 'ref': d['ref']+'/caption-body'})
                    continue
        result.append(d)
    labels=[r for r in case_rows if r['score']>=.9 and case_pattern.fullmatch(r['text'].strip())]
    for i,row in enumerate(labels):
        box=fitz.Rect(row['bbox'])
        peers=[fitz.Rect(r['bbox']) for r in labels if abs(r['bbox'][0]-box.x0)<page.rect.width*.06]
        if len(peers)>=3:
            width=float(np.median([b.width for b in peers]));height=float(np.median([b.height for b in peers]))
            # An occasional detector box includes the adjacent circuit despite
            # recognizing only its label. Constrain such an outlier using the
            # repeated labels in that same column, not the graph's pixels.
            if box.width>width*1.5 or box.height>height*1.5:
                center=(box.y0+box.y1)/2;left=float(np.median([b.x0 for b in peers]))
                box=fitz.Rect(left,center-height/2,left+width,center+height/2)
        text='Case '+re.search(r'\d+',row['text'])[0]
        result.append({'role':'text','bbox':box,'text':text,'ref':f'raster/case/{i}'})
    return result
