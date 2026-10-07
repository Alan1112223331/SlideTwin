"""Repair MuPDF-generated supplementary ToUnicode scalars as UTF-16BE.

Some MuPDF text writers emit `<1d456>` instead of `<d835dc56>` in a PDF
ToUnicode CMap. PDF strings contain UTF-16BE, not scalar code points. These
invalid strings can silently decode into an unrelated BMP character.

Only caller-specified newly inserted fonts are eligible. Corrected CMaps are
cloned before attaching them to those fonts, preserving original-page fonts
even if their old CMap was shared with an inserted font.
"""
from __future__ import annotations

import re


HEX = rb'<([0-9A-Fa-f]+)>'
BLOCK = re.compile(rb'\b(\d+)\s+begin(bfchar|bfrange)\b(.*?)\bend\2\b', re.S)
CHAR = re.compile(HEX+rb'\s*'+HEX)
RANGE = re.compile(HEX+rb'\s*'+HEX+rb'\s*(?:'+HEX+rb'|\[([^\]]*)\])', re.S)
MAX_RANGE = 65536


def scalar_string(raw):
    """Accept only the known malformed 5/6-hex-digit scalar encoding."""
    if len(raw) not in (5, 6):
        return None
    value = int(raw, 16)
    return value if 0x10000 <= value <= 0x10FFFF else None


def utf16(value):
    return chr(value).encode('utf-16-be').hex().encode('ascii')


def repair_cmap(stream: bytes) -> tuple[bytes, dict]:
    """Repair only bfchar destinations and bfrange destinations/arrays.

    Source CIDs, codespace ranges, valid UTF-16 strings, names, comments and
    unrelated PDF hex strings remain byte-identical. A malformed oversized
    range is left intact and counted rather than expanded without bounds.
    """
    stats = {'bfchar_destinations': 0, 'bfrange_destinations': 0, 'ranges_skipped': 0}
    remaining = MAX_RANGE

    def char_match(match):
        nonlocal remaining
        source, target = match.groups()
        value = scalar_string(target)
        if value is None:
            return match[0]
        if remaining <= 0:
            return match[0]
        remaining -= 1
        stats['bfchar_destinations'] += 1
        return b'<'+source+b'> <'+utf16(value)+b'>'

    def range_match(match):
        nonlocal remaining
        start, end, target, array = match.groups()
        first, last = int(start,16), int(end,16)
        count = last-first+1
        if not 0 <= first <= last <= 65535 or count > remaining:
            stats['ranges_skipped'] += 1
            return match[0]
        if target is not None:
            value = scalar_string(target)
            if value is None:
                return match[0]
            if count <= 0 or count > MAX_RANGE or value+count-1 > 0x10FFFF:
                stats['ranges_skipped'] += 1
                return match[0]
            # An explicit UTF-16 array crosses low-surrogate boundaries safely.
            values = b' '.join(b'<'+utf16(value+offset)+b'>' for offset in range(count))
            stats['bfrange_destinations'] += count
            remaining -= count
            return b'<'+start+b'> <'+end+b'> ['+values+b']'

        changed = 0
        if len(re.findall(HEX, array)) != count:
            stats['ranges_skipped'] += 1
            return match[0]
        def array_match(item):
            nonlocal changed
            value = scalar_string(item[1])
            if value is None:
                return item[0]
            changed += 1
            return b'<'+utf16(value)+b'>'
        replaced = re.sub(HEX, array_match, array)
        if not changed:
            return match[0]
        stats['bfrange_destinations'] += changed
        remaining -= changed
        return b'<'+start+b'> <'+end+b'> ['+replaced+b']'

    def block_match(match):
        count, kind, body = match.groups()
        # Do not interpret comment text as mappings.
        parts = re.split(rb'(%[^\r\n]*)', body)
        pattern, callback = (CHAR, char_match) if kind == b'bfchar' else (RANGE, range_match)
        repaired = b''.join(part if part.startswith(b'%') else pattern.sub(callback, part) for part in parts)
        return match[0] if repaired == body else match[0].replace(body, repaired, 1)

    repaired = BLOCK.sub(block_match, stream)
    return repaired, stats


def repair_story_font_unicode(pdf, font_xrefs) -> dict:
    """Fix only explicitly selected inserted Font objects, before subsetting.

    Returns diagnostics counts; leaves valid fonts untouched. Cloning CMaps
    guarantees that an original font sharing a map cannot be modified through
    this operation. It never reads or writes content-stream text or glyphs.
    """
    stats = {'fonts_repaired': 0, 'cmaps_repaired': 0, 'bfchar_destinations': 0,
             'bfrange_destinations': 0, 'ranges_skipped': 0, 'fonts_skipped': 0}
    maps = {}
    for ref in sorted(set(font_xrefs)):
        if not isinstance(ref, int) or not 0 < ref < pdf.xref_length():
            stats['fonts_skipped'] += 1
            continue
        if pdf.xref_get_key(ref, 'Type') != ('name', '/Font'):
            stats['fonts_skipped'] += 1
            continue
        kind, value = pdf.xref_get_key(ref, 'ToUnicode')
        if kind != 'xref':
            continue
        map_ref = int(value.split()[0])
        if map_ref not in maps:
            try:
                original = pdf.xref_stream(map_ref)
            except (RuntimeError, ValueError):
                maps[map_ref] = None
                stats['fonts_skipped'] += 1
                continue
            if original is None:
                maps[map_ref] = None
                continue
            fixed, details = repair_cmap(original)
            for key, count in details.items():
                stats[key] += count
            if fixed == original:
                maps[map_ref] = None
                continue
            replacement = pdf.get_new_xref()
            pdf.update_object(replacement, pdf.xref_object(map_ref))
            pdf.update_stream(replacement, fixed)
            maps[map_ref] = replacement
            stats['cmaps_repaired'] += 1
        replacement = maps[map_ref]
        if replacement is not None:
            pdf.xref_set_key(ref, 'ToUnicode', f'{replacement} 0 R')
            stats['fonts_repaired'] += 1
    return stats
