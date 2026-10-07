"""Build-time only: reproducible, static TrueType SC fonts for MuPDF Story."""
from io import BytesIO
import hashlib
from pathlib import Path
from urllib.request import urlopen
import unicodedata
from html import escape
import argparse
import copy

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BASE = 'https://raw.githubusercontent.com/google/fonts/a85815a42757630ce188fdad368c2dfc444d4773/ofl/notosanssc/'
FILES = {
    'NotoSansSC%5Bwght%5D.ttf': 'a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da',
    'OFL.txt': '1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9',
}

def repair_font_maps(font):
    """Change cmap aliases and ten unsafe digit substitutions, never outlines.

    Noto Sans SC's `locl` lookup selects unmapped ASCII-digit alternates in
    mixed script runs such as Q1. MuPDF then cannot build the correct reverse
    Unicode map. Keep default digit glyphs for those substitutions only; the
    rest of GSUB, including CJK localization, remains intact.
    """
    unicode_maps = [table for table in font['cmap'].tables if table.isUnicode()]
    for table in unicode_maps:
        for code, glyph in list(table.cmap.items()):
            if 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F:
                canonical = unicodedata.normalize('NFKC', chr(code))
                if len(canonical) == 1 and ord(canonical) != code and table.cmap.get(ord(canonical)) == glyph:
                    del table.cmap[code]
    # ASCII hyphen, soft hyphen, and nonbreaking hyphen share one glyph in
    # this font. Preserve every character and its outline, but give aliases
    # independent glyph IDs so PDF reverse maps cannot turn '-' into U+2011.
    if 'glyf' in font and 'hmtx' in font:
        best = font.getBestCmap()
        hyphen = best.get(0x2D)
        for code in (0xAD, 0x2011):
            if hyphen and best.get(code) == hyphen:
                alias = f'{hyphen}.unicode{code:04X}'
                if alias not in font.getGlyphOrder():
                    font['glyf'][alias] = copy.deepcopy(font['glyf'][hyphen])
                    font['hmtx'].metrics[alias] = font['hmtx'].metrics[hyphen]
                    if 'vmtx' in font:
                        font['vmtx'].metrics[alias] = font['vmtx'].metrics[hyphen]
                    order = font.getGlyphOrder()
                    font.setGlyphOrder(order if alias in order else [*order, alias])
                for table in unicode_maps:
                    if table.cmap.get(code) == hyphen:
                        table.cmap[code] = alias
    reverse = {}
    for table in unicode_maps:
        for code, glyph in table.cmap.items():
            reverse.setdefault(glyph, set()).add(code)
    digits = {glyph for glyph, codes in reverse.items() if len(codes) == 1 and 48 <= next(iter(codes)) <= 57}
    removed = []
    if 'GSUB' not in font:
        return removed
    gsub = font['GSUB'].table
    indexes = {index for feature in gsub.FeatureList.FeatureRecord if feature.FeatureTag == 'locl'
               for index in feature.Feature.LookupListIndex}
    for index in sorted(indexes):
        lookup = gsub.LookupList.Lookup[index]
        for sub in lookup.SubTable:
            sub_type = lookup.LookupType
            if sub_type == 7:
                sub_type, sub = sub.ExtensionLookupType, sub.ExtSubTable
            if sub_type != 1:
                continue
            for source, target in list(sub.mapping.items()):
                if source in digits and target not in reverse:
                    del sub.mapping[source]
                    removed.append({'lookup': index, 'unicode': next(iter(reverse[source])), 'source': source, 'target': target})
    return removed


def validate_font_roundtrip(fonts):
    """Mandatory build check, both before and after PDF font subsetting/reopen."""
    import pymupdf as fitz
    from slidetwin.render import css_fonts, explicit_fallbacks, unicode_font, math_font
    from slidetwin.font_unicode import repair_story_font_unicode
    css, archive = css_fonts(tuple(fonts))
    samples = [
        '逻辑电路 锁存器 利用上下文 Q1 Q2 Q3 EE6316 0123456789 日期26Nov2026',
        ''.join(chr(code) for code in range(33, 127)),
        'Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 26Nov2026 1.25 3/4',
        '目标：10⁶ TFLOP/s；𝑥 ∈ ℝⁿ',
    ]
    expected = []
    with fitz.open() as pdf:
        for family in ('twin,latin', 'latin,twin'):
            for bold in (False, True):
                page = pdf.new_page(width=900, height=650)
                weight = 'bold' if bold else 'normal'
                for index, text in enumerate(samples):
                    body=escape(text)
                    if index==3:
                        faces=[(name,fitz.Font(fontfile=str(path))) for name,path in [('math',math_font()),('unicode',unicode_font())] if path]
                        body=explicit_fallbacks(body,fitz.Font(fontfile=str(fonts[int(bold)])),faces)
                    result = page.insert_htmlbox(fitz.Rect(20, 20+index*130, 880, 140+index*130),
                        f'<div style="font-family:{family};font-weight:{weight};font-size:16pt">{body}</div>',
                        css=css, archive=archive)
                    assert result[0] >= 0, 'Font round-trip sample did not fit'
                expected.append(samples)
        repair_story_font_unicode(pdf,{font[0] for page in pdf for font in page.get_fonts(full=True)})
        for page, texts in zip(pdf, expected):
            actual = ''.join(page.get_text().split())
            assert all(''.join(text.split()) in actual for text in texts), 'Font round-trip failed before subset'
        pdf.subset_fonts()
        content = pdf.tobytes(garbage=4, deflate=True)
    with fitz.open(stream=content, filetype='pdf') as reopened:
        for page, texts in zip(reopened, expected):
            actual = ''.join(page.get_text().split())
            assert all(''.join(text.split()) in actual for text in texts), 'Font round-trip failed after subset/reopen'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file', type=Path, help='Optional local cache of the checksum-pinned variable font')
    parser.add_argument('--destination', type=Path, default=Path('/usr/share/fonts/truetype/slidetwin'))
    args = parser.parse_args()
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        data = args.source_file.read_bytes() if name.endswith('.ttf') and args.source_file else urlopen(BASE+name, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Pinned font download checksum mismatch')
        if name.endswith('.ttf'):
            for style, weight in [('Regular', 400), ('Bold', 700)]:
                font = instantiateVariableFont(TTFont(BytesIO(data)), {'wght': weight}, inplace=True)
                removed = repair_font_maps(font)
                if len(removed) != 10 or {entry['unicode'] for entry in removed} != set(range(48,58)):
                    raise RuntimeError('Pinned font digit substitution layout changed; review required')
                font.save(destination/f'NotoSansSC-{style}.ttf')
        else:
            (destination/name).write_bytes(data)
    validate_font_roundtrip((destination/'NotoSansSC-Regular.ttf', destination/'NotoSansSC-Bold.ttf'))


if __name__ == '__main__':
    main()
