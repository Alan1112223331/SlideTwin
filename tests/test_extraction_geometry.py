"""Regressions for source ownership and mathematical geometry (no model calls)."""

import pymupdf as fitz

from slidetwin.extract import enrich_page, native_lines, native_visual_lines, restore


def descriptor(text, box, role="text", ref="r"):
    return {"text": text, "bbox": fitz.Rect(box), "role": role, "ref": ref}


def test_formula_prefix_and_quantifier_share_one_translation_owner():
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 70), "where K = x", fontsize=18)
        regions, _ = enrich_page(page, 1, [descriptor("", [28, 48, 180, 80], "formula")])
    assert [r.source for r in regions] == ["where K ="]
    owned = [tuple(box) for region in regions for box in region.erase]
    assert len(owned) == len(set(owned))


def test_parent_table_descriptor_with_equivalent_dash_is_not_an_ocr_target():
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        labels = [(50, 50, "Two-class"), (45, 66, "Classification"), (50, 82, "Verification")]
        ds = []
        for x, y, text in labels:
            page.insert_text((x, y), text, fontsize=12)
            ds.append(descriptor(text, [x-1, y-12, x+100, y+3], ref=text))
        page.insert_text((80, 98), "-", fontsize=12)
        ds.append(descriptor("-", [79, 86, 86, 101], ref="dash"))
        # Docling's larger table cell covers the already extracted native
        # labels. It uses a different dash encoding than the PDF glyph.
        ds.append(descriptor("Two-class Classification Verification –", [44, 38, 145, 102], "table_cell", "parent"))
        regions, issues = enrich_page(page, 1, ds)
    assert regions and all(r.native for r in regions)
    assert any(issue["kind"] == "duplicate_native_ocr" and issue["text"].endswith("–") for issue in issues)
    assert "Two-class" in " ".join(r.source for r in regions)


def test_distinct_raster_label_is_not_suppressed_by_nearby_native_text():
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        page.insert_text((30, 60), "Native label", fontsize=12)
        regions, _ = enrich_page(page, 1, [descriptor("Image label", [160, 48, 250, 64], ref="image")])
    assert any(not r.native and r.source == "Image label" for r in regions)
    assert any(r.native and r.source == "Native label" for r in regions)


def test_stacked_scripts_and_following_prose_stay_in_one_region():
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=180)
        prefix = "Profile compares "
        page.insert_text((30, 70), prefix, fontsize=11)
        x = 30+fitz.get_text_length(prefix, fontsize=11)
        atoms = []
        for subscript in ("1", "2"):
            page.insert_text((x, 70), "x", fontname="heit", fontsize=11)
            sx = x+fitz.get_text_length("x", fontname="heit", fontsize=11)
            page.insert_text((sx, 66), "3", fontsize=8)
            page.insert_text((sx, 74), subscript, fontsize=8)
            atoms.append((x, sx))
            x = sx+fitz.get_text_length(subscript, fontsize=8)+2
            suffix = " and " if subscript == "1" else " values."
            page.insert_text((x, 70), suffix, fontsize=11)
            x += fitz.get_text_length(suffix, fontsize=11)
        page.insert_text((30, 86), "The paragraph continues here.", fontsize=11)
        ds = [descriptor("Profile compares x 3 1 and x 3 2 values. The paragraph continues here.", [28, 52, 365, 92]),
              # A tighter detection encloses only the first subscript. It
              # cannot steal that glyph from the mathematical atom.
              descriptor("1", [atoms[0][1]-.1, 68, atoms[0][1]+5, 76], ref="small")]
        regions, _ = enrich_page(page, 1, ds)
        original = native_lines(page)
        reconstructed = native_visual_lines(original)
    assert len(reconstructed) == 2
    assert len(regions) == 1
    region = regions[0]
    assert "Profile compares" in region.source and "paragraph continues" in region.source
    assets = list(region.protected_assets.values())
    assert len(assets) == 2
    assert sorted(asset["text"] for asset in assets) == ["x31", "x32"]
    for asset in assets:
        assert asset["bbox"][1] < 66 and asset["bbox"][3] > 74
    assert " and " in restore(region, region.source)


def test_visual_line_merge_does_not_cross_parallel_columns():
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=160)
        page.insert_text((25, 60), "Left label", fontsize=12)
        page.insert_text((250, 60), "Right label", fontsize=12)
        lines = native_visual_lines(native_lines(page))
        regions, _ = enrich_page(page, 1, [])
    assert len(lines) == 2
    assert [r.source for r in regions] == ["Left label", "Right label"]


def test_separate_pdf_objects_keep_measured_word_gap_inside_one_descriptor():
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=160)
        page.insert_text((30, 60), "Types", fontsize=12)
        x = 30+fitz.get_text_length("Types", fontsize=12)+8
        page.insert_text((x, 60), "Polynomials", fontsize=12)
        regions, _ = enrich_page(page, 1, [descriptor("Types Polynomials", [28, 44, 180, 64], "page_header")])
    assert len(regions) == 1
    assert regions[0].source == "Types Polynomials"


def test_rotated_native_text_keeps_original_glyph_reading_order():
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=160)
        page.insert_text((250, 80), "Power density", fontsize=12, rotate=180)
        original = native_lines(page)
        reconstructed = native_visual_lines(original)
    assert "".join(c["c"] for c in reconstructed[0]["chars"]) == "Power density"
    assert reconstructed[0]["dir"] == original[0]["dir"]


def test_math_asset_does_not_absorb_adjacent_ordinary_font_prose():
    # The original failure turned "r_DS is" into one untranslatable image.
    from slidetwin.extract import protect_native_math

    def char(c, x, baseline, size, font):
        return {"c": c, "origin": (x, baseline), "size": size, "font": font,
                "bbox": (x, baseline-size*.8, x+size*.5, baseline+size*.2)}

    chars = [char("r", 10, 40, 18, "CambriaMath"),
             char("D", 19, 44, 12, "CambriaMath"), char("S", 25, 44, 12, "CambriaMath"),
             char("i", 35, 40, 18, "Arial"), char("s", 40, 40, 18, "Arial")]
    source, _, assets = protect_native_math(chars, "")
    assert source.endswith("is")
    assert [asset["text"] for asset in assets.values()] == ["rDS"]


def test_tex_math_font_glyph_uses_the_original_visual_asset():
    from slidetwin.extract import protect_native_math

    chars = [{"c": "λ", "origin": (10, 40), "size": 12, "font": "rtxmi",
              "bbox": (10, 30, 17, 43)},
             {"c": " ", "origin": (17, 40), "size": 12, "font": "TeXGyreTermes-Regular",
              "bbox": (17, 30, 20, 43)},
             {"c": "v", "origin": (20, 40), "size": 12, "font": "TeXGyreTermes-Regular",
              "bbox": (20, 30, 26, 43)}]
    source, _, assets = protect_native_math(chars, "")
    assert source.endswith(" v")
    assert [asset["text"] for asset in assets.values()] == ["λ"]


def test_mathematical_alphanumeric_encoding_protects_opaque_font_aliases():
    from slidetwin.extract import protect_native_math

    def glyph(c,x,y=40,font='F2'):
        return {'c':c,'origin':(x,y),'size':16,'font':font,
                'bbox':(x,y-12,x+8,y+3)}
    # Equal-size glyphs defeat the size-based script detector. The explicit
    # Unicode mathematical encoding must still retain the original positions.
    chars=[glyph(c,10+i*8) for i,c in enumerate('Voltage ')]
    chars.extend([glyph('𝑣',74),glyph('𝐺',82,44),glyph('𝑆',90,44)])
    chars.extend(glyph(c,98+i*8) for i,c in enumerate(' is high'))
    source,protected,assets=protect_native_math(chars,'')
    assert source=='Voltage ⟦P000⟧ is high'
    assert protected=={'⟦P000⟧':'𝑣𝐺𝑆'}
    assert assets['⟦P000⟧']['bbox']==[74,28,98,47]
    assert assets['⟦P000⟧']['baseline_down']==7


def test_ordinary_greek_and_latin_prose_are_not_encoded_math_assets():
    from slidetwin.extract import protect_native_math
    text='αβ language'
    chars=[{'c':c,'origin':(10+i*6,40),'size':12,'font':'F2',
            'bbox':(10+i*6,30,16+i*6,43)} for i,c in enumerate(text)]
    source,protected,assets=protect_native_math(chars,'')
    assert source==text and protected=={} and assets=={}


def test_geometry_version_refresh_reuses_verified_docling_cache(tmp_path, monkeypatch):
    from slidetwin.extract import EXTRACT_VERSION, extract
    from slidetwin.models import Document, Page, digest, read_cache, write_json
    from slidetwin import formula_ocr, raster_refine
    import sys

    source = tmp_path/"source.pdf"
    work = tmp_path/"work"
    work.mkdir()
    with fitz.open() as pdf:
        page = pdf.new_page(width=250, height=120)
        page.insert_text((30, 60), "Cached source label", fontsize=14)
        pdf.save(source)
    sha = digest(source.read_bytes())
    Document(sha, [Page(1, 250, 120, [])]).save(work/"document.json")
    write_json(work/"docling-document.json", {"texts": [], "tables": []})
    write_json(work/"extraction-key.json", {
        "key": digest(sha+"33"+str([1])), "source_sha256": sha,
        "selected_pages": [1], "extract_version": "33",
        "docling_sha256": digest((work/"docling-document.json").read_bytes()),
        "document_sha256": digest((work/"document.json").read_bytes()),
        "complete": True, "docling_complete": True,
    })
    # A native-geometry refresh must not instantiate the ML converter.
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    monkeypatch.setattr(formula_ocr, "supplement", lambda *args: [])
    monkeypatch.setattr(raster_refine, "refine", lambda page, descriptors, work: descriptors)
    messages = []
    document = extract(source, work, log=messages.append)
    assert document.pages[0].regions[0].source == "Cached source label"
    assert any("Reusing Docling object tree" in message for message in messages)
    assert read_cache(work/"extraction-key.json")["extract_version"] == EXTRACT_VERSION
