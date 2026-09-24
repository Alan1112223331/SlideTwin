"""Build-time only: reproducible, static TrueType SC fonts for MuPDF Story."""
from io import BytesIO
import hashlib
from pathlib import Path
from urllib.request import urlopen
import unicodedata

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BASE = 'https://raw.githubusercontent.com/google/fonts/a85815a42757630ce188fdad368c2dfc444d4773/ofl/notosanssc/'
FILES = {
    'NotoSansSC%5Bwght%5D.ttf': 'a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da',
    'OFL.txt': '1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9',
}
destination = Path('/usr/share/fonts/truetype/slidetwin')
destination.mkdir(parents=True, exist_ok=True)
for name, expected in FILES.items():
    data = urlopen(BASE + name, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Pinned font download checksum mismatch')
    if name.endswith('.ttf'):
        for style, weight in [('Regular', 400), ('Bold', 700)]:
            font = instantiateVariableFont(TTFont(BytesIO(data)), {'wght': weight}, inplace=True)
            # MuPDF's reverse glyph map otherwise selects CJK compatibility
            # aliases (e.g. U+F937) instead of their standard Unicode character.
            # Remove only duplicate aliases pointing at the identical glyph;
            # this changes the font map, never model text or document semantics.
            for table in font['cmap'].tables:
                if not table.isUnicode():
                    continue
                for code, glyph in list(table.cmap.items()):
                    if 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F:
                        canonical = unicodedata.normalize('NFKC', chr(code))
                        if len(canonical) == 1 and ord(canonical) != code and table.cmap.get(ord(canonical)) == glyph:
                            del table.cmap[code]
            font.save(destination / f'NotoSansSC-{style}.ttf')
    else:
        (destination / name).write_bytes(data)

# A build must not silently reintroduce glyph aliases or missing Chinese text.
import pymupdf as fitz
from slidetwin.render import css_fonts
css, archive = css_fonts((destination / 'NotoSansSC-Regular.ttf', destination / 'NotoSansSC-Bold.ttf'))
with fitz.open() as pdf:
    page = pdf.new_page()
    text = '逻辑电路 锁存器 利用上下文'
    page.insert_htmlbox(fitz.Rect(20, 20, 550, 150), '<div style="font-family:twin">' + text + '</div>', css=css, archive=archive)
    assert ''.join(text.split()) in ''.join(page.get_text().split()), 'Chinese font round-trip failed'
