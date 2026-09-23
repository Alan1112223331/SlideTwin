from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path
import json
import re
import statistics

import pymupdf as fitz

from .models import Document, Page, Region, digest, write_json, read_cache
from .raster import raster_container


EXTRACT_VERSION = "33"
BULLETS = "•●▪◦‣–➢\uf0b7\uf0a7\uf071\uf06d✓✔☑❑"
MATH_FONT = re.compile(r"symbol|math|cmmi|cmsy|cmex|mtextra", re.I)
NUMBERS = re.compile(r"(?<![\w⟦])\d+(?:[.,]\d+)*(?:%|[⁰¹²³⁴⁵⁶⁷⁸⁹])?(?![\w⟧])")
SYMBOLIC_AXIS = re.compile(r"[IVPRCLftxyz][a-z₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹_]{0,3}\s*\(\s*(?:[fpnuμµmkMGT]?(?:A|V|W|s|m|F|H|Ω|Hz)|dB|K|°C)\s*\)")
IMMUTABLE_VARIABLE = re.compile(r'(?:t(?:phl|plh|p?ff\d*|su\d*|h\d*|hold|skew|cq|cl|cs|cycle|jitter|pd|cd|inv|xor|and|or|clock|clk|ctk|ox|i|p|min|max)(?:[-_]?(?:min|max))?|v(?:in|out|gs|ds|gd|dd|ss|th|tp|tn|sb|bs|t)|i(?:ds|d|s|g)|c(?:gd|gs|gb|ox|in|out)|g(?:[sdmb]|gd)|[dp](?:in|out)|su|clk|er|ec)',re.I)
MATH_FUNCTIONS={'clip','round','min','max','abs','sign','relu','softmax','sigmoid','sqrt','floor','ceil','sin','cos','tan','tanh','exp','log','ln','sum','median','mean','var','std'}


def immutable_math_label(text: str) -> bool:
    value=re.sub(r'⟦P\d+⟧','0',text).strip()
    if value in {'-min','ds','dsat','ox','SiO','SiO2','permicon','gsol','gdol','ref'}:return True
    if re.fullmatch(r'(?:jbssw[g]?|jSWG|JSWG)',value,re.I):return True
    value=value.lstrip('→←↑↓ ')
    if re.fullmatch(r'(?:C[_{]?(?:sb|db)[}]?|V(?:reg|rev|ref))',value,re.I):return True
    if SYMBOLIC_AXIS.fullmatch(value):return True
    if re.fullmatch(r'C\s+(?:metal|gate)',value):return True
    if re.fullmatch(r't\s+(?:setup|hold)',value,re.I):return True
    if re.fullmatch(r'(?:softmax|sigmoid|relu)\([A-Z0-9 +*/^().-]+',value):return True
    if IMMUTABLE_VARIABLE.fullmatch(value):return True
    if re.search('[=<>≤≥+−-]',value):
        words=re.findall(r'[A-Za-z]+',value)
        symbolic=lambda w: (len(w)==1 or w in MATH_FUNCTIONS or IMMUTABLE_VARIABLE.fullmatch(w)
                            or re.fullmatch(r'(?:d[xVy]|t(?:su|h)(?:min|max)R|t(?:Logic|interconnect))',w,re.I)
                            or (w.isupper() and len(w)<=4))
        if words and all(symbolic(w) for w in words):return True
    if not re.search(r'\d',value):return False
    words=re.findall(r'[A-Za-zμµΩ]+',value)
    units=re.compile(r'(?:[fpnumkMGTμµ]?(?:s|Hz|V|A|W|F|H|m|Ω)|sec)')
    return bool(words) and all(units.fullmatch(w) or IMMUTABLE_VARIABLE.fullmatch(w) for w in words)


def rect(bbox: dict, height: float) -> fitz.Rect:
    if str(bbox.get("coord_origin", "TOPLEFT")).upper().endswith("BOTTOMLEFT"):
        return fitz.Rect(bbox["l"], height-bbox["t"], bbox["r"], height-bbox["b"])
    return fitz.Rect(bbox["l"], bbox["t"], bbox["r"], bbox["b"])


def needs_translation(text: str) -> bool:
    clean = re.sub(r"https?://\S+|\S+@\S+|⟦P\d+⟧", "", text)
    words = re.findall(r"[A-Za-z]{2,}", clean)
    if not words:
        return False
    if re.fullmatch(r"[A-Za-z]{1,6}\([^)]*\)", clean.strip()):
        return False
    if re.search(r"[._^]", clean) and not re.search(r"[A-Za-z]{4,}", clean) and " " not in clean.strip():
        return False
    # Pure equations, identifiers and numeric page numbers remain original objects.
    if re.search(r"[=∑∫√]", clean) and not re.search(r"[A-Za-z]{4,}", clean) and not re.match(r"In\s+(?:\d{4}|⟦P\d+⟧)",text):
        return False
    return True


def protect(text: str) -> tuple[str, dict[str, str]]:
    values = {}
    # Program-defined tokens, never model-generated positioning or HTML.
    pattern = re.compile(r"https?://\S+|\S+@\S+|\$[^$]+\$|\\\(.*?\\\)|\b[A-Za-z][₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹]+|(?<!\w)[A-Za-z]+_[A-Za-z0-9{}]+|(?<!\w)\d+(?=(?:st|nd|rd|th)\b)|(?<!\w)\d+(?:[.,]\d+)*(?:%|[⁰¹²³⁴⁵⁶⁷⁸⁹])?(?!\w)")
    def replace(match):
        key = f"⟦P{len(values):03d}⟧"
        values[key] = match.group()
        return key
    return pattern.sub(replace, text), values


def restore(region: Region, text: str) -> str:
    for key, value in region.protected.items():
        text = text.replace(key, value)
    return text


def ordered_cell_chars(chars: list[dict]) -> list[dict]:
    """Anchor visual rows with main-size glyphs before assigning subscripts."""
    rows = []
    for char in sorted(chars, key=lambda c: -c['size']):
        candidates = [row for row in rows if abs(row[0]['origin'][1]-char['origin'][1]) < row[0]['size']*.6]
        if candidates:
            min(candidates, key=lambda row: abs(row[0]['origin'][1]-char['origin'][1])).append(char)
        else:
            rows.append([char])
    result = []
    for row in sorted(rows, key=lambda row: row[0]['origin'][1]):
        row.sort(key=lambda c: c['origin'][0])
        if result:
            result.append({**row[0], 'c':' ', 'font':'spacing'})
        result.extend(row)
    return result


def protect_native_math(chars: list[dict], text: str) -> tuple[str, dict[str, str], dict[str, dict]]:
    """Keep mathematical font glyphs as inline visual assets, not guessed Unicode.

    The model sees immutable tokens. Only the program can crop and place these
    source glyphs; this also preserves native superscripts and subscripts.
    """
    expanded = []
    for char in chars:
        if expanded:
            previous = expanded[-1]
            if char["origin"][0] < previous["origin"][0] and char["origin"][1]-previous["origin"][1] > char["size"]*0.6:
                expanded.append({**char, "c": " ", "font": "spacing"})
        expanded.append(char)
    chars = expanded
    # Superscripts/subscripts may use an ordinary font rather than a math font.
    # Compare only nearby glyphs on the same visual line, not another paragraph.
    mask = []
    for c in chars:
        nearby = [p for p in chars if abs(p["origin"][1]-c["origin"][1]) < max(p["size"], c["size"])*0.7 and p["size"] > c["size"]*1.18]
        shifted = any(abs(p["origin"][1]-c["origin"][1]) > p["size"]*0.12 for p in nearby)
        mask.append((bool(MATH_FONT.search(c["font"])) or shifted) and not c["c"].isspace())
    # Author identities are immutable source glyphs too. This prevents a model
    # from omitting a given name or inconsistently transliterating a repeated
    # copyright footer across pages.
    raw_text = "".join(c["c"] for c in chars)
    prose_indices=set()
    for match in re.finditer(r'[\uf000-\uf0ffα-ωΑ-Ω](is|are|and|the)\b',raw_text):
        prose_indices.update(range(match.start(1),match.end(1)))
    # English ordinal suffixes are prose, not a mathematical sub/superscript.
    # Preserve the number as a normal token so "1st" can become "第1个".
    for match in re.finditer(r'(?<!\w)\d+(?:st|nd|rd|th)\b',raw_text):
        for j in range(match.start(),match.end()):mask[j]=False
    for match in re.finditer(r"(?:©\s*)?(?:Dr\.|Prof\.)\s+(?:[A-Z][a-z]+|[A-Z]\.)(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){0,4}(?:\s*\|)?", raw_text):
        for j in range(match.start(), match.end()):
            mask[j] = True
    for i, is_math in enumerate(list(mask)):
        if not is_math:
            continue
        for step in [-1, 1]:
            j = i+step
            while 0 <= j < len(chars) and j not in prose_indices and re.fullmatch(r"[A-Za-z0-9α-ωΑ-Ω₀-₉ₐ-ₜ⁰¹²³⁴⁵⁶⁷⁸⁹_]", chars[j]["c"]):
                mask[j] = True
                j += step
    runs = []
    pieces = []
    i = 0
    while i < len(chars):
        if not mask[i]:
            pieces.append(chars[i]["c"])
            i += 1
            continue
        start = i
        while i < len(chars) and mask[i]:
            i += 1
        pieces.append(f"⟪M{len(runs)}⟫")
        runs.append(chars[start:i])
    text = re.sub(r"\s+", " ", "".join(pieces)).strip()
    assets_by_marker = {}
    for index, run in enumerate(runs):
        value = "".join(c["c"] for c in run)
        if not value.strip():
            continue
        box = fitz.Rect(run[0]["bbox"])
        for c in run[1:]:
            box |= fitz.Rect(c["bbox"])
        main_size=max(c['size'] for c in run)
        row_baseline=statistics.median(c['origin'][1] for c in run if c['size']>=main_size*.85)
        nearby=[c['origin'][1] for c in chars if not MATH_FONT.search(c['font']) and c['size']>=main_size*.85 and abs(c['origin'][1]-row_baseline)<main_size*.3]
        baseline=statistics.median(nearby) if nearby else row_baseline
        marker = f"⟪M{index}⟫"
        assets_by_marker[marker] = {"text": value, "bbox": list(box), "baseline_down": max(0, box.y1-baseline)}
    text, protected = protect(text)
    assets = {}
    for marker, asset in assets_by_marker.items():
        key = f"⟦P{len(protected):03d}⟧"
        text = text.replace(marker, key)
        protected[key] = asset["text"]
        assets[key] = asset
    return text, protected, assets


def inline_styles(chars: list[dict], base: tuple) -> list[dict]:
    runs = []
    for c in chars:
        style = (c["color"], bool(c["flags"] & 16), bool(c["flags"] & 2))
        if not runs or runs[-1][0] != style:
            runs.append((style, []))
        previous = runs[-1][1][-1] if runs[-1][1] else None
        if previous and c["origin"][0] < previous["origin"][0] and c["origin"][1]-previous["origin"][1] > c["size"]*0.6:
            runs[-1][1].append({**c, "c": " "})
        runs[-1][1].append(c)
    result = []
    for style, run in runs:
        phrase = re.sub(r"\s+", " ", "".join(c["c"] for c in run)).strip()
        if style == base or not needs_translation(phrase) or not re.search(r"[A-Za-z]{3,}", phrase):
            continue
        # Mathematical runs are handled by immutable original-glyph assets.
        if any(MATH_FONT.search(c["font"]) for c in run) or re.search(r"\b(?:Dr|Prof)\.", phrase):
            continue
        result.append({"source": phrase, "color": style[0], "bold": style[1], "italic": style[2]})
    return result


def native_lines(page: fitz.Page) -> list[dict]:
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = []
            for span in line["spans"]:
                for char in span["chars"]:
                    chars.append({**char, "size": span["size"], "font": span["font"], "color": span["color"], "flags": span["flags"]})
            if chars:
                lines.append({"chars": chars, "dir": list(line["dir"])})
    # Some slide exporters draw a second, gray copy of each word as its shadow.
    # Keep one semantic glyph, but retain both boxes for eventual redaction.
    # Matching is local and character-by-character; repeated words elsewhere are
    # never collapsed. Different colors plus near-identical geometry are required.
    chars = [c for line in lines for c in line['chars']]
    buckets=defaultdict(list)
    for index,char in enumerate(chars):buckets[(char['c'].casefold(),char['font'],round(char['size'],1))].append((index,char))
    removed = set()
    for index,shadow in enumerate(chars):
        rgb = [(shadow['color'] >> shift) & 255 for shift in (16,8,0)]
        gray_shadow=max(rgb)-min(rgb)<=2 and 96<=rgb[0]<=225
        box=fitz.Rect(shadow['bbox']);limit=min(3,shadow['size']*.13)
        matches=[]
        for position,foreground in buckets[(shadow['c'].casefold(),shadow['font'],round(shadow['size'],1))]:
            if foreground is shadow or id(foreground) in removed:continue
            identical=foreground['color']==shadow['color'] and foreground['c']==shadow['c'] and position<index
            if not identical and not gray_shadow:continue
            if abs(foreground['size']-shadow['size'])>.1:continue
            fg=[(foreground['color']>>shift)&255 for shift in (16,8,0)]
            if not identical and sum(fg)>=sum(rgb):continue
            other=fitz.Rect(foreground['bbox'])
            if max(abs(a-b) for a,b in zip(box,other))<=limit:
                matches.append(foreground)
        if matches:
            foreground=min(matches,key=lambda c:sum(abs(a-b) for a,b in zip(box,c['bbox'])))
            foreground.setdefault('shadow_boxes',[]).append(list(box));removed.add(id(shadow))
    return [{**line,'chars':[c for c in line['chars'] if id(c) not in removed]} for line in lines
            if any(id(c) not in removed for c in line['chars'])]


def native_bullet(char: dict) -> bool:
    return char['c'] in BULLETS and not ('\ue000'<=char['c']<='\uf8ff' and MATH_FONT.search(char['font']))


def native_subscript_line(chars: list[dict], native: list[dict]) -> bool:
    literal=''.join(c['c'] for c in chars).strip()
    if not re.fullmatch(r'[a-z]{3,}',literal):return False
    first=chars[0]
    return any(c['size']>first['size']*1.18 and c['c'].isalnum()
               and -first['size']*.35 < first['bbox'][0]-c['bbox'][2] < first['size']*.4
               and first['size']*.1 < first['origin'][1]-c['origin'][1] < first['size']
               for other in native for c in other['chars'])


def _descriptors(raw: dict, heights: dict[int, float]) -> dict[int, list[dict]]:
    pages = defaultdict(list)
    for item in raw.get("texts", []):
        for prov in item.get("prov", []):
            page = int(prov["page_no"])
            pages[page].append({"ref": item["self_ref"], "role": item.get("label", "text"),
                                "bbox": rect(prov["bbox"], heights[page]), "text": item.get("text", "")})
    for table in raw.get("tables", []):
        if not table.get("prov"):
            continue
        page = int(table["prov"][0]["page_no"])
        for i, cell in enumerate(table["data"]["table_cells"]):
            if not cell.get("text", "").strip() or not cell.get("bbox"):
                continue
            pages[page].append({"ref": f'{table["self_ref"]}/cell/{i}', "role": "table_cell",
                                "bbox": rect(cell["bbox"], heights[page]), "text": cell["text"]})
    return pages


def repeated_brand_marks(pdf: fitz.Document, descriptors: dict, selected_map: dict[int, int]) -> dict[int, list[fitz.Rect]]:
    """Keep repeated university logos as artwork, including their OCR lettering.

    Require identical small margin artwork on at least three pages and an
    institution word inside the image. Body diagrams and one-off figures do not
    qualify. The source image is never erased or replaced.
    """
    occurrences = defaultdict(list)
    for index, page in enumerate(pdf):
        for item in page.get_image_info(hashes=True):
            box = fitz.Rect(item["bbox"])
            margin = box.y1 < page.rect.height*0.2 or box.y0 > page.rect.height*0.8
            if margin and box.get_area() < page.rect.get_area()*0.10:
                occurrences[item["digest"]].append((index+1, box))
    result = defaultdict(list)
    for entries in occurrences.values():
        repeated = len({n for n, box in entries}) >= 3
        words = []
        for number, box in entries:
            if number in selected_map:
                words.extend(d["text"] for d in descriptors[selected_map[number]] if d["bbox"] in box + (-1, -1, 1, 1))
        label = " ".join(words)
        if not re.search(r"\b(?:university|college|institute)\b|大学|学院", label, re.I):
            continue
        if not repeated and not (len(label.split()) >= 3 and label.upper() == label):
            continue
        for number, box in entries:
            result[number].append(box)
    return result


def complete_page_context(page: fitz.Page, descriptors: list[dict]) -> str:
    """Retain all native text and add only Docling text not already present there.

    Matching is local to the descriptor's rectangle, so a word elsewhere on the
    page does not accidentally suppress text inside an image or diagram.
    """
    # PDF's sort=True inserts large horizontal padding to imitate page columns.
    # Geometry is stored separately; that padding adds tokens, not meaning.
    context = re.sub(r"[ \t]+", " ", page.get_text(sort=True))
    context = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", context)
    extra = []
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    for descriptor in descriptors:
        text = normalize(descriptor["text"])
        if not text:
            continue
        native = normalize(page.get_textbox(descriptor["bbox"] + (-1, -1, 1, 1)))
        compact_text = re.sub(r"\s+", "", text)
        compact_native = re.sub(r"\s+", "", native)
        if compact_text == compact_native or (compact_text and compact_text in compact_native):
            continue
        extra.append(descriptor["text"])
    if extra:
        context += "\nAdditional text from document structure and figures:\n"+"\n".join(extra)
    return context


def enrich_page(page: fitz.Page, number: int, descriptors: list[dict], preserved_artwork: list[fitz.Rect] | None = None) -> tuple[list[Region], list[dict]]:
    """Docling supplies structure; PDF glyph geometry supplies exact indents/styles.

    Assign every native glyph at most once. Recover text missed by Docling rather
    than silently dropping diagram labels. Never reflow a formula-only object.
    """
    groups = defaultdict(list)
    issues = []
    native = native_lines(page)
    # Recover a word's final glyph when a PDF exporter emits it as a separate
    # text object. Join only an adjacent same-font, same-baseline alphabetic run.
    consumed=set()
    for line in native:
        chars=line['chars']
        if not re.fullmatch(r'[A-Za-z]{3,}', ''.join(c['c'] for c in chars).strip()):continue
        last=chars[-1]
        for other in native:
            tail=other['chars']
            if other is line or id(other) in consumed or len(tail)>2:continue
            if (all(c['c'].isalpha() for c in tail) and tail[0]['font']==last['font']
                    and abs(tail[0]['size']-last['size'])<.1
                    and abs(tail[0]['origin'][1]-last['origin'][1])<.1
                    and -last['size']*.08<=tail[0]['bbox'][0]-last['bbox'][2]<last['size']*.12):
                chars.extend(tail);consumed.add(id(other))
    native=[line for line in native if id(line) not in consumed]
    for line_index, line in enumerate(native):
        raw_line = "".join(c["c"] for c in line["chars"]).strip()
        preserve_axis = line["dir"] != [1.0, 0.0] and bool(SYMBOLIC_AXIS.fullmatch(raw_line))
        assigned = defaultdict(list)
        for sequence,ch in enumerate(line["chars"]):
            box = fitz.Rect(ch["bbox"])
            center = (box.tl + box.br) / 2
            choices = [(i, d) for i, d in enumerate(descriptors) if center in d["bbox"] + (-1, -1, 1, 1)]
            # A precise cell/text region wins over an oversized parent region.
            index = min(choices, key=lambda pair: pair[1]["bbox"].get_area())[0] if choices else -line_index-1
            assigned[index].append({**ch,'sequence':sequence})
        # A descriptor boundary cannot divide a contiguous alphabetic word.
        literal=''.join(c['c'] for c in line['chars'])
        for word in re.finditer(r'[A-Za-z]{3,}',literal):
            owners=[(i,c) for i,chars in assigned.items() for c in chars if word.start()<=c['sequence']<word.end()]
            if len({i for i,c in owners})<2:continue
            winner=Counter(i for i,c in owners).most_common(1)[0][0]
            for i,c in owners:
                if i!=winner:assigned[i].remove(c);assigned[winner].append(c)
        assigned={i:sorted(chars,key=lambda c:c['sequence']) for i,chars in assigned.items() if chars}
        for index, chars in assigned.items():
            spaced=[]
            for char in chars:
                if spaced:
                    previous=spaced[-1]['sequence']
                    gap=line['chars'][previous+1:char['sequence']]
                    # A superscript space can fall outside the Docling box while
                    # its neighboring words fall inside. Retain that real space.
                    if gap and all(c['c'].isspace() for c in gap):
                        spaced.append({**gap[0],'sequence':previous+1})
                spaced.append(char)
            chars=spaced
            groups[index].append({"chars": chars, "dir": line["dir"], "line": line_index, "preserve_axis": preserve_axis})
    entries = []
    matched = set()
    for index, lines in groups.items():
        descriptor = descriptors[index] if index >= 0 else {"role": "native_fallback", "ref": f"native/{-index}"}
        if index >= 0:
            matched.add(index)
        # Symbols with SI units are original mathematical artwork, including
        # rotated axes. Do not send fragments such as 'ds' or '(mA)' to an LLM.
        lines = [line for line in lines if not line["preserve_axis"] and
                 not immutable_math_label(''.join(c['c'] for c in line['chars']).strip()) and
                 not native_subscript_line(line['chars'],native)]
        if not lines:
            continue
        if descriptor["role"] == "code":
            literal=' '.join(''.join(c['c'] for c in line['chars']) for line in lines)
            # Docling occasionally calls worked arithmetic a code block. Its
            # natural-language units and explanations still require translation.
            arithmetic=(re.search(r'\b(?:values|bytes|storage)\b',literal,re.I)
                        and re.search(r'\d\s*[×÷]',literal)
                        and not re.search(r'[{};]|\b(?:def|return|import|printf|function)\b',literal))
            if not arithmetic:continue
            descriptor={**descriptor,'role':'text'}
        if descriptor['role']=='formula':
            # A displayed equation may include an English label before a colon.
            # Translate that label and leave all equation glyphs as source objects.
            labeled=[]
            for line in lines:
                chars=line['chars'];colon=next((i for i,c in enumerate(chars) if c['c']==':'),None)
                literal=''.join(c['c'] for c in chars)
                if literal.strip() in {'channel','overlap'}:continue
                # An English subscript is part of the formula, not a diagram
                # caption. Recognize its larger, immediately preceding base.
                first=chars[0]
                if re.fullmatch(r'[a-z]{3,}',literal.strip()) and any(
                    c['size']>first['size']*1.18 and c['c'].isalnum()
                    and -first['size']*.3 < first['bbox'][0]-c['bbox'][2] < first['size']*.4
                    and first['origin'][1]-c['origin'][1]>first['size']*.1
                    and first['origin'][1]-c['origin'][1]<first['size']
                    for other in native for c in other['chars']):continue
                # Quantifiers inside an equation are prose; equation operands
                # remain source-owned artwork on either side of the phrase.
                for phrase in re.finditer(r'\b(?:for every|for all|where)\b',literal,re.I):
                    labeled.append({**line,'chars':chars[phrase.start():phrase.end()]})
                if colon is not None:
                    prefix=''.join(c['c'] for c in chars[:colon+1])
                    if needs_translation(prefix) and re.search(r'[A-Za-z]{4,}',prefix):
                        labeled.append({**line,'chars':chars[:colon+1]})
                elif any(w.lower() not in MATH_FUNCTIONS and not IMMUTABLE_VARIABLE.fullmatch(w)
                         for w in re.findall(r'[A-Za-z]{4,}',''.join(c['c'] for c in chars))):
                    equals=next((i for i,c in enumerate(chars) if c['c']=='='),None)
                    prefix=''.join(c['c'] for c in chars[:equals]) if equals is not None else ''
                    labeled.append({**line,'chars':chars[:equals+1] if equals is not None and re.search(r'[A-Za-z]{4,}',prefix) else chars})
            lines=labeled
            if not lines:continue
        reversed_rows=any(a['chars'][0]['origin'][1]>b['chars'][0]['origin'][1]+a['chars'][0]['size']*.3 for a,b in zip(lines,lines[1:]))
        lines.sort(key=lambda x: (min(c['bbox'][1] for c in x['chars']), x['chars'][0]['origin'][0])
                   if descriptor['role'] in {'title','section_header','formula'} or reversed_rows else (x['line'], 0))
        # Beamer numbered badges can be a separate PDF line at the same visual
        # baseline as their list text. Keep the badge and its digit at their
        # original coordinates; translate only the body starting after the gap.
        if descriptor["role"] == "list_item" and len(lines) >= 2:
            marker, body = lines[0]["chars"], lines[1]["chars"]
            marker_text = "".join(c["c"] for c in marker).strip()
            marker_box = fitz.Rect(marker[0]["bbox"])
            for c in marker[1:]:
                marker_box |= fitz.Rect(c["bbox"])
            body_box = fitz.Rect(body[0]["bbox"])
            if (re.fullmatch(r"\(?\d{1,3}[.)]?", marker_text)
                    and body_box.y0 < marker_box.y1 and body_box.y1 > marker_box.y0
                    and body_box.x0-marker_box.x1 > max(2, body[0]["size"]*0.5)):
                lines = lines[1:]
        # Split lists at each source bullet and indentation/column changes. A line
        # wrapped by the source retains the first line's actual hanging indent.
        chunks = []
        for line in lines:
            chars = line["chars"]
            text = "".join(c["c"] for c in chars).strip()
            if not text:
                continue
            leading=next((c for c in chars if not c['c'].isspace()),chars[0])
            new_bullet = native_bullet(leading)
            previous = chunks[-1][-1] if chunks else None
            new_column = previous and abs(chars[0]["bbox"][0]-previous["chars"][0]["bbox"][0]) > max(40, 3*chars[0]["size"])
            gap = previous and chars[0]["bbox"][1]-previous["chars"][0]["bbox"][1] > 2.1*chars[0]["size"]
            def has_left_key(row):
                first=row['chars'][0]
                return any('=' in ''.join(c['c'] for c in other['chars']) and
                           max(c['bbox'][2] for c in other['chars'])<first['bbox'][0]-3 and
                           abs(other['chars'][0]['origin'][1]-first['origin'][1])<first['size']*.4
                           for other in native)
            keyed_row=previous and has_left_key(previous) and has_left_key(line)
            independent_label=previous and (descriptor['role']=='formula' or reversed_rows and descriptor['role']=='text') and len(text)<45
            heading_row=previous and descriptor['role'] in {'title','section_header'} and abs(chars[0]['size']-previous['chars'][0]['size'])>max(chars[0]['size'],previous['chars'][0]['size'])*.2
            if not chunks or new_bullet or new_column or gap or keyed_row or independent_label or heading_row:
                chunks.append([])
            chunks[-1].append(line)
        for chunk in chunks:
            allchars = [c for line in chunk for c in line["chars"]]
            # The original bullet remains untouched. Never invent a fixed indent.
            while allchars and (allchars[0]["c"].isspace() or native_bullet(allchars[0])):
                allchars.pop(0)
            while allchars and allchars[-1]["c"].isspace():
                allchars.pop()
            if not allchars:
                continue
            text = " ".join("".join(c["c"] for c in line["chars"] if c in allchars).strip() for line in chunk).strip()
            text = re.sub(r"\s+", " ", text)
            if not needs_translation(text):
                continue
            author=(page.parent.metadata or {}).get('author','').strip()
            if author and text==author:
                continue
            boxes = [fitz.Rect(c["bbox"]) for c in allchars if not c["c"].isspace()]
            box = fitz.Rect(boxes[0])
            for b in boxes[1:]:
                box |= b
            sizes = [c["size"] for c in allchars if c["c"].isalnum()]
            size = statistics.median(sizes or [allchars[0]["size"]])
            base = Counter((c["color"], bool(c["flags"] & 16), bool(c["flags"] & 2)) for c in allchars if not c["c"].isspace()).most_common(1)[0][0]
            protected_source, protected, assets = protect_native_math(allchars, text)
            if protected and not needs_translation(protected_source):
                continue
            if re.fullmatch(r"⟦P\d+⟧", protected_source) and any(re.search(r"\b(?:Dr|Prof)\.",a["text"]) for a in assets.values()):
                continue
            # Redact each native glyph, not a broad Docling box. Formulas, bullets,
            # rules, background and adjacent labels remain outside the edit mask.
            erase = [list(b + (0.015, 0.015, -0.015, -0.015)) for b in boxes]
            erase.extend(b for c in allchars for b in c.get('shadow_boxes',[]))
            entries.append(Region(id="", page=number, source=protected_source, bbox=list(box), role=descriptor["role"],
                                  size=size, color=base[0], bold=base[1],
                                  erase=erase, protected=protected, protected_assets=assets, inline_styles=inline_styles(allchars, base), direction=chunk[0]["dir"], docling_ref=descriptor["ref"]))
    for index, descriptor in enumerate(descriptors):
        if index in matched or descriptor["role"] in {"formula", "code"} or not needs_translation(descriptor["text"]) or immutable_math_label(descriptor['text']):
            continue
        if any(descriptor["bbox"] in box + (-1, -1, 1, 1) for box in (preserved_artwork or [])):
            issues.append({"page": number, "kind": "preserved_brand_artwork", "blocking": False,
                           "text": descriptor["text"], "bbox": list(descriptor["bbox"])})
            continue
        # Docling OCR can rediscover native diagram labels with a shorter box
        # that misses the font's ascender. Match glyph centers against its vertical
        # span and local horizontal bounds, instead of requiring full containment.
        b=descriptor['bbox']
        local=[c for line in native for c in line['chars'] if
               b.x0-2 <= (c['bbox'][0]+c['bbox'][2])/2 <= b.x1+2 and
               max(b.y0,c['bbox'][1]) < min(b.y1,c['bbox'][3]) and
               abs((b.y0+b.y1)/2-(c['bbox'][1]+c['bbox'][3])/2)<max(b.height,c['size'])*.5]
        local.sort(key=lambda c:(round(c['origin'][1]/max(1,c['size']*.4)),c['origin'][0]))
        normalized=lambda value:re.sub(r'\s+','',value).casefold()
        if normalized(descriptor['text']) and normalized(descriptor['text'])==normalized(''.join(c['c'] for c in local)):
            issues.append({'page':number,'kind':'duplicate_native_ocr','blocking':False,'text':descriptor['text'],'bbox':list(b)})
            continue
        # OCR labels carry the actual OCR region. Raster treatment is separately
        # gated by the renderer's uniform-background check.
        source, protected = protect(descriptor["text"])
        b = descriptor["bbox"]
        entries.append(Region(id="", page=number, source=source, bbox=list(b), role=descriptor["role"], native=False,
                              size=max(5, min(b.height*0.78, 18)), protected=protected, docling_ref=descriptor["ref"]))
    # Docling may emit the same bitmap label twice, including a truncated OCR
    # duplicate. Require both geometric overlap and identical/prefix text.
    ocr=sorted([r for r in entries if not r.native],key=lambda r:len(restore(r,r.source)),reverse=True)
    retained=[]
    for r in ocr:
        literal=re.sub(r'\s+','',restore(r,r.source)).casefold();box=fitz.Rect(r.bbox)
        duplicate=any(re.sub(r'\s+','',restore(other,other.source)).casefold().startswith(literal) and
                      (box & fitz.Rect(other.bbox)).get_area()>.8*min(box.get_area(),fitz.Rect(other.bbox).get_area())
                      for other in retained)
        if duplicate:entries.remove(r)
        else:retained.append(r)
    # OCR often splits one chart sentence at every visual line. Translate the
    # complete cell as one semantic unit; line wrapping remains program-owned.
    cells=defaultdict(list)
    for entry in entries:
        if entry.native:continue
        cell=raster_container(page,fitz.Rect(entry.bbox))
        if cell is not None:cells[tuple(round(v,1) for v in cell)].append(entry)
    for group in cells.values():
        if len(group)<2:continue
        group.sort(key=lambda r:(round(r.bbox[1],1),r.bbox[0]))
        # A cell can contain parallel columns; only merge successive text lines.
        if any(b.bbox[1]<a.bbox[3]-min(a.size,b.size)*.5 for a,b in zip(group,group[1:])):continue
        text=' '.join(restore(r,r.source) for r in group)
        source,protected=protect(text)
        box=fitz.Rect(group[0].bbox)
        for entry in group[1:]:box|=fitz.Rect(entry.bbox)
        for entry in group:entries.remove(entry)
        entries.append(Region('',number,source,list(box),native=False,size=max(r.size for r in group),
                              protected=protected,docling_ref=';'.join(r.docling_ref for r in group)))
    # A native table cell may be fragmented by overlapping Docling descriptors
    # and PDF span boundaries. Recover the complete sentence in original glyph
    # reading order before translation, keeping the cell's measured typography.
    vector_cells=defaultdict(list)
    rectangles=[fitz.Rect(d['rect']) for d in page.get_drawings() if d.get('color') is not None and
                len(d['items'])==1 and d['items'][0][0]=='re']
    for entry in entries:
        if not entry.native:continue
        candidates=[b for b in rectangles if fitz.Rect(entry.bbox) in b+(-2,-2,2,2) and
                    b.height<entry.size*4 and b.width>entry.size*3]
        if candidates:vector_cells[tuple(min(candidates,key=lambda b:b.get_area()))].append(entry)
    headings=sorted([r for r in entries if r.native and r.role in {'title','section_header'}],key=lambda r:(r.bbox[1],r.bbox[0]))
    for left,right in zip(headings,headings[1:]):
        if abs(left.bbox[1]-right.bbox[1])<.5 and abs(left.size-right.size)<.5 and 0<=right.bbox[0]-left.bbox[2]<left.size*.8:
            if not any(left in group or right in group for group in vector_cells.values()):
                vector_cells[tuple(fitz.Rect(left.bbox)|fitz.Rect(right.bbox))]=[left,right]
    # A full-width sentence can continue in a shorter indented object on the
    # next baseline. Treat that continuation as one translation unit, so Chinese
    # word order cannot push the main clause into the tiny second-line object.
    prose=sorted([r for r in entries if r.native and r.role not in {'title','section_header','page_footer'}],
                 key=lambda r:(r.bbox[1],r.bbox[0]))
    for first,following in zip(prose,prose[1:]):
        a,b=restore(first,first.source),restore(following,following.source)
        centered_tail=(re.fullmatch(r'[a-z]{2,12}[.;]?',b) and
                       first.bbox[2]-first.bbox[0]>.7*page.rect.width and
                       abs(sum(first.bbox[::2])-sum(following.bbox[::2]))<first.size*2)
        indented_tail=(first.bbox[2]>.9*page.rect.width and
                       first.bbox[0]<=following.bbox[0]<first.bbox[0]+6*first.size)
        if (len(a)>75 and not re.search(r'[.;:!?]$',a) and re.match(r'[a-z]',b)
                and (indented_tail or centered_tail)
                and .65*first.size<following.bbox[1]-first.bbox[1]<1.5*first.size
                and abs(first.size-following.size)<.5
                and not any(first in group or following in group for group in vector_cells.values())):
            combined=fitz.Rect(first.bbox)|fitz.Rect(following.bbox)
            others=[r for r in entries if r not in (first,following) and (fitz.Rect(r.bbox).tl+fitz.Rect(r.bbox).br)/2 in combined]
            if all(r.native and re.match(r'[({\[]',restore(r,r.source)) and abs(r.bbox[1]-following.bbox[1])<first.size*.4 for r in others):
                vector_cells[tuple(combined)]=[first,following,*others]
    for coordinates,group in vector_cells.items():
        if len(group)<2:continue
        cell=fitz.Rect(coordinates)
        if any(r.direction!=group[0].direction for r in group):continue
        ordered=sorted(group,key=lambda r:r.bbox[0])
        if any(abs(a.bbox[1]-b.bbox[1])<max(a.size,b.size)*.6 and
               b.bbox[0]-a.bbox[2]>max(a.size,b.size)*1.5 for a,b in zip(ordered,ordered[1:])):continue
        owned=[fitz.Rect(b)+(-.03,-.03,.03,.03) for r in group for b in r.erase]
        chars=[c for line in native for c in line['chars'] if
               any((fitz.Rect(c['bbox']).tl+fitz.Rect(c['bbox']).br)/2 in b for b in owned)
               or (c['c'].isspace() and (fitz.Rect(c['bbox']).tl+fitz.Rect(c['bbox']).br)/2 in cell)]
        chars=ordered_cell_chars(chars)
        text=re.sub(r'\s+',' ',''.join(c['c'] for c in chars)).strip()
        if not text:continue
        source,protected,assets=protect_native_math(chars,text)
        base=Counter((c['color'],bool(c['flags']&16),bool(c['flags']&2)) for c in chars if not c['c'].isspace()).most_common(1)[0][0]
        box=fitz.Rect(group[0].bbox)
        for r in group[1:]:box.include_rect(fitz.Rect(r.bbox))
        for r in group:entries.remove(r)
        entries.append(Region('',number,source,list(box),role='table_cell',size=statistics.median(r.size for r in group),
                              color=base[0],bold=base[1],erase=[b for r in group for b in r.erase],protected=protected,
                              protected_assets=assets,inline_styles=inline_styles(chars,base),direction=group[0].direction,
                              docling_ref=';'.join(r.docling_ref for r in group)))
    entries.sort(key=lambda r: (round(r.bbox[1], 1), r.bbox[0]))
    for index, entry in enumerate(entries):
        entry.id = f"p{number:04d}_r{index:04d}"
        if abs(entry.direction[0]-1) > 0.02 or abs(entry.direction[1]) > 0.02:
            issues.append({"page": number, "kind": "rotated_text", "id": entry.id, "blocking": True})
    return entries, issues


def extract(source: Path, work: Path, selected: list[int] | None = None, force=False, log=print) -> Document:
    sha = digest(source.read_bytes())
    work.mkdir(parents=True, exist_ok=True)
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open(source) as original:
        selected = selected or list(range(1, len(original)+1))
        if len(set(selected)) != len(selected) or any(p < 1 or p > len(original) for p in selected):
            raise ValueError("Invalid/duplicate page selection")
        key = digest(sha + EXTRACT_VERSION + str(selected))
        cache = work / "extraction-key.json"
        if not force and cache.exists() and (work/"document.json").exists():
            stored = read_cache(cache)
            if stored.get("key") == key and stored.get('complete',True) and stored.get("document_sha256") == digest((work/"document.json").read_bytes()):
                try:document=Document.load(work/'document.json')
                except (ValueError,TypeError,KeyError):log('Invalid extraction cache; rebuilding this checkpoint')
                else:
                    log("Reusing verified extraction cache")
                    return document
        subset_path = work / "selected-source.pdf"
        with fitz.open() as subset:
            for number in selected:
                subset.insert_pdf(original, from_page=number-1, to_page=number-1)
            subset.save(subset_path, garbage=4, deflate=True)
        previous = read_cache(cache)
        raw_path = work/"docling-document.json"
        reuse_raw = (not force and raw_path.exists() and previous.get('docling_complete',True) and previous.get("source_sha256") == sha and previous.get("selected_pages") == selected)
        if reuse_raw and previous.get("docling_sha256"):
            reuse_raw = digest(raw_path.read_bytes()) == previous["docling_sha256"]
        raw=read_cache(raw_path) if reuse_raw else {}
        reuse_raw=bool(reuse_raw and raw)
        docling_complete=True
        docling_failed_pages=[]
        if not reuse_raw:
            # A verified cache hit needs only PyMuPDF. Avoid importing the ML
            # runtime on every resumed translation or native-geometry refresh.
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            opts = PdfPipelineOptions()
            opts.do_ocr = True
            opts.do_table_structure = True
            opts.table_structure_options.do_cell_matching = True
            converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            log(f"Docling: parsing {len(selected)} selected pages; retaining all {len(original)} pages as translation context")
            result = converter.convert(subset_path,raises_on_error=False)
            status=str(result.status).split(".")[-1].lower()
            if status not in {'success','partial_success'}:
                raise RuntimeError("Docling conversion did not complete successfully")
            docling_complete=status=='success'
            if not docling_complete:
                log('Docling returned partial results; retaining usable pages and native text')
                failures=getattr(result,'errors',[])
                failed_numbers={getattr(e,'page_no',None) for e in failures}
                if not failed_numbers or None in failed_numbers:docling_failed_pages=list(selected)
                else:docling_failed_pages=[selected[n-1] for n in failed_numbers if isinstance(n,int) and 1<=n<=len(selected)] or list(selected)
            result.document.save_as_json(raw_path)
            write_json(work/'ocr-provenance.json',{'source_sha256':sha,'selected_pages':selected,
                       'pages':{str(selected[parsed.page_no-1]):any(getattr(cell,'from_ocr',False) for cell in parsed.cells)
                                for parsed in result.pages if 1<=parsed.page_no<=len(selected)}})
        else:
            log("Reusing Docling object tree; refreshing native geometry and protected content")
        raw = read_cache(raw_path)
        heights = {i+1: original[n-1].rect.height for i, n in enumerate(selected)}
        descriptors = _descriptors(raw, heights)
        selected_map = {n: i+1 for i, n in enumerate(selected)}
        brand_marks = repeated_brand_marks(original, descriptors, selected_map)
        ocr_records={}
        ocr_path=work/'ocr-provenance.json'
        if ocr_path.exists():
            saved_ocr=read_cache(ocr_path)
            if saved_ocr.get('source_sha256')==sha and saved_ocr.get('selected_pages')==selected:
                ocr_records=saved_ocr.get('docling_cell_flags',saved_ocr.get('pages',{}))
        pages, issues = [], []
        if not docling_complete:
            issues.extend({'page':n,'kind':'docling_partial_result','blocking':True} for n in docling_failed_pages)
        for index, page in enumerate(original):
            number = index+1
            regions = [];extra=[]
            if number in selected_map:
                from .formula_ocr import supplement
                from .raster_refine import refine
                original_descriptors=descriptors.get(selected_map[number],[])
                try:extra=supplement(page,original_descriptors,work)
                except (RuntimeError,ValueError) as exc:
                    issues.append({'page':number,'kind':'supplemental_ocr_failed','reason':str(exc),'blocking':True})
                refined=original_descriptors+extra
                try:refined=refine(page,refined,work)
                except (RuntimeError,ValueError) as exc:
                    issues.append({'page':number,'kind':'ocr_refinement_failed','reason':str(exc),'blocking':True})
                try:regions, page_issues = enrich_page(page, number,refined, brand_marks.get(number))
                except (RuntimeError,ValueError) as exc:
                    issues.append({'page':number,'kind':'page_enrichment_failed','reason':str(exc),'blocking':True})
                    try:regions,page_issues=enrich_page(page,number,[],brand_marks.get(number))
                    except (RuntimeError,ValueError) as fallback_exc:
                        regions=[]
                        page_issues=[{'page':number,'kind':'page_text_unavailable','reason':str(fallback_exc),'blocking':True}]
                issues.extend(page_issues)
            context = page.get_text(sort=True)
            if number in selected_map:
                context = complete_page_context(page, descriptors.get(selected_map[number],[]))
            ocr_used=bool(ocr_records.get(str(number),False) or extra or any(not r.native for r in regions))
            pages.append(Page(number, page.rect.width, page.rect.height, regions, context,ocr_used=ocr_used))
        document = Document(sha, pages, issues)
        document.save(work/"document.json")
        write_json(ocr_path,{'source_sha256':sha,'selected_pages':selected,
                   'pages':{str(p.number):p.ocr_used for p in pages if p.number in selected},
                   'docling_cell_flags':ocr_records,
                   'method':'Docling from_ocr flags OR supplemental OCR results OR non-native OCR text regions. Cell flags may be unavailable after Docling unloads a page.'})
        write_json(cache, {"key": key, "selected_pages": selected, "source_sha256": sha, "extract_version": EXTRACT_VERSION,
                           'docling_complete':docling_complete,'complete':docling_complete and not any(i.get('kind') in {'docling_partial_result','supplemental_ocr_failed','ocr_refinement_failed','page_enrichment_failed'} for i in issues),
                           "docling_sha256": digest(raw_path.read_bytes()), "document_sha256": digest((work/"document.json").read_bytes())})
        return document
