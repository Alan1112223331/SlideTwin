"""Supplementary math/emoji CMaps use UTF-16, without touching source fonts."""
import pymupdf as fitz

from slidetwin.font_unicode import repair_cmap, repair_story_font_unicode


def cmap(body):
    return b'/CIDInit /ProcSet findresource begin\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n'+body


def test_repairs_destinations_only_and_preserves_legal_utf16_and_comments():
    raw = cmap(b'4\tbeginbfchar\n<0001> <1d456>\n<1d456> <0041>\n<0003> <d835dc56>\n<0004> <00410042>\n'
               b'% <0005> <1f603> comment untouched\nendbfchar\n/unrelated <1f603>')
    fixed, stats = repair_cmap(raw)
    assert b'<0001> <d835dc56>' in fixed
    assert b'<1d456> <0041>' in fixed
    assert b'<0003> <d835dc56>' in fixed and b'<0004> <00410042>' in fixed
    assert b'% <0005> <1f603> comment untouched' in fixed and fixed.endswith(b'/unrelated <1f603>')
    assert stats['bfchar_destinations'] == 1 and stats['bfrange_destinations'] == 0
    assert repair_cmap(fixed)[0] == fixed


def test_range_expansion_handles_low_surrogate_boundary_and_array_values():
    raw = cmap(b'3 beginbfrange\n<0001> <0003> <103ff>\n'
               b'<0004> <0006> [<1d456> <0041> <1f603>]\n'
               b'<0007> <0008> <0042>\nendbfrange')
    fixed, stats = repair_cmap(raw)
    assert b'<0001> <0003> [<d800dfff> <d801dc00> <d801dc01>]' in fixed
    assert b'<0004> <0006> [<d835dc56> <0041> <d83dde03>]' in fixed
    assert b'<0007> <0008> <0042>' in fixed
    assert stats['bfrange_destinations'] == 5


def test_invalid_and_oversized_ranges_are_retained_instead_of_expanded():
    raw = cmap(b'4 beginbfrange\n<0001> <fffff> <1d456>\n'
               b'<0003> <0002> <1d456>\n<0004> <0005> [<1d456>]\n'
               b'<0006> <0007> <10ffff>\nendbfrange')
    fixed, stats = repair_cmap(raw)
    assert fixed == raw and stats['ranges_skipped'] == 4


def test_valid_map_remains_byte_identical_including_header_spacing():
    raw = cmap(b'1\t\tbeginbfchar\n<0001> <d835dc56>\nendbfchar\n'
               b'1  beginbfrange\n<0002> <0003> <0041>\nendbfrange')
    assert repair_cmap(raw)[0] == raw


def test_cloned_map_does_not_change_unselected_original_font():
    with fitz.open() as pdf:
        pdf.new_page().insert_text((20,40), 'Original source')
        original_font = pdf[0].get_fonts()[0][0]
        shared = pdf.get_new_xref()
        raw = cmap(b'1 beginbfchar\n<0001> <1d456>\nendbfchar')
        pdf.update_object(shared, '<<>>')
        pdf.update_stream(shared, raw)
        pdf.xref_set_key(original_font, 'ToUnicode', f'{shared} 0 R')
        inserted_font = pdf.get_new_xref()
        pdf.update_object(inserted_font, pdf.xref_object(original_font))
        original_object = pdf.xref_object(original_font)
        stats = repair_story_font_unicode(pdf, [inserted_font])
        assert stats['fonts_repaired'] == 1
        assert pdf.xref_object(original_font) == original_object and pdf.xref_stream(shared) == raw
        replacement = int(pdf.xref_get_key(inserted_font,'ToUnicode')[1].split()[0])
        assert replacement != shared and b'<d835dc56>' in pdf.xref_stream(replacement)


def test_unrecognized_font_selector_is_skipped_without_blocking_other_repairs():
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_htmlbox(fitz.Rect(20,20,500,150), 'Math 𝑥')
        refs = [font[0] for font in page.get_fonts(full=True)]
        stats = repair_story_font_unicode(pdf, [0, pdf.xref_length()+1, page.xref, *refs])
        assert stats['fonts_skipped'] == 3 and stats['fonts_repaired'] > 0


def test_real_story_math_emoji_bold_subset_save_reopen_and_source_untouched():
    sample = 'Math 𝑥𝑦𝑧 = 𝑓(𝑥), 𝛼 + 𝛽 ≤ 𝜎² 𝔸 𝕏 👍 😃'
    with fitz.open() as pdf:
        source = pdf.new_page()
        source.insert_text((20,40), 'Original page 0123456789', fontname='helv')
        source_font_refs = {font[0] for font in source.get_fonts(full=True)}
        source_fonts = {ref: pdf.xref_object(ref) for ref in source_font_refs}
        source_pix = source.get_pixmap().samples
        # Story's built-in mathematical and emoji fallback reproduces the
        # encoding failure on ordinary installations, without a font download.
        inserted_fonts = set()
        for bold in (False, True):
            page = pdf.new_page(width=900, height=180)
            page.insert_htmlbox(fitz.Rect(20,20,880,150),
                f'<div style="font-family:sans-serif;font-weight:{700 if bold else 400}">{sample}</div>')
            inserted_fonts.update(font[0] for font in page.get_fonts(full=True))
        inserted_fonts -= source_font_refs
        stats = repair_story_font_unicode(pdf, inserted_fonts)
        assert stats['fonts_repaired'] > 0
        assert {ref:pdf.xref_object(ref) for ref in source_font_refs} == source_fonts
        assert pdf[0].get_pixmap().samples == source_pix
        before_subset = pdf.tobytes()
        pdf.subset_fonts()
        after_subset = pdf.tobytes(garbage=4, deflate=True)
    for serialized in (before_subset, after_subset):
        with fitz.open(stream=serialized, filetype='pdf') as reopened:
            assert reopened[0].get_text().strip() == 'Original page 0123456789'
            assert reopened[0].get_pixmap().samples == source_pix
            for page in list(reopened)[1:]:
                assert ''.join(sample.split()) in ''.join(page.get_text().split())
