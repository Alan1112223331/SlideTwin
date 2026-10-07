from pathlib import Path
import io

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from slidetwin.config import Layout
from slidetwin.extract import native_lines
from slidetwin.models import Document, Page, Region, digest
from slidetwin.render import (build_plan, content_html, css_fonts,
                              native_parenthesized_asset_box)


def original_inline_math(tmp_path):
    """Two source assets with independent colors and a lowered subscript."""
    fitz.TOOLS.set_small_glyph_heights(True)
    source=tmp_path/'original.pdf'
    pdf=fitz.open();page=pdf.new_page(width=240,height=120)
    size=16;baseline=45
    x=20
    for text,fontsize,y,color in [('Input (',size,baseline,(0,0,0)),
                                  ('x',size,baseline,(1,0,0)),
                                  ('s',10,baseline+5,(0,0,1)),
                                  (') output',size,baseline,(0,0,0))]:
        page.insert_text((x,y),text,fontsize=fontsize,color=color)
        x+=fitz.get_text_length(text,fontsize=fontsize)
    lines=native_lines(page)
    chars=[c for line in lines for c in line['chars']]
    box=fitz.Rect(chars[0]['bbox'])
    for char in chars[1:]:box.include_rect(fitz.Rect(char['bbox']))
    region=Region('r',1,'Input (⟦P000⟧⟦P001⟧) output',list(box),size=size,
                  erase=[c['bbox'] for c in chars],
                  protected={'⟦P000⟧':'x','⟦P001⟧':'s'})
    images=[]
    for token,literal in region.protected.items():
        char=next(c for c in chars if c['c']==literal)
        region.protected_assets[token]={'text':literal,'bbox':char['bbox'],
                                        'baseline_down':char['bbox'][3]-char['origin'][1]}
        images.append({'text':literal,'bbox':char['bbox'],'baseline':char['origin'][1]})
    pdf.save(source)
    return pdf,page,source,region,lines,images


def test_two_native_math_assets_and_brackets_remain_atomic_in_narrow_pdf(tmp_path):
    pdf,page,source,region,lines,images=original_inline_math(tmp_path)
    expected_box,down=native_parenthesized_asset_box(region,lines,images)
    document=Document(digest(source.read_bytes()),[Page(1,240,120,[region])])
    layout=Layout()
    plan,issues=build_plan(source,document,{'r':'Translated prefix （⟦P000⟧⟦P001⟧） translated suffix'},
                           [1],layout,tmp_path/'work')
    assert not issues and len(plan)==1
    entry=plan[0]
    assert entry.html_body.count('<img ')==1
    assert entry.extractable_text=='Translated prefix  translated suffix'
    assert len(entry.assets)==1
    image_path=Path(next(iter(entry.assets.values())))
    expected=page.get_pixmap(clip=expected_box,dpi=360,alpha=True)
    pixels=np.asarray(Image.open(image_path).convert('RGBA'))
    expected_pixels=np.asarray(Image.open(io.BytesIO(expected.tobytes('png'))).convert('RGBA'))
    assert np.array_equal(pixels,expected_pixels)
    # Both source colors survive composition, including the lowered blue s.
    assert np.count_nonzero((pixels[:,:,0]>180)&(pixels[:,:,1]<80)&(pixels[:,:,2]<80))>10
    assert np.count_nonzero((pixels[:,:,2]>180)&(pixels[:,:,0]<80)&(pixels[:,:,1]<80))>10
    css,archive=css_fonts(layout.fonts())
    for name,path in entry.assets.items():archive.add((Path(path).read_bytes(),name))
    narrow=fitz.open();out=narrow.new_page(width=120,height=240)
    spare,scale=out.insert_htmlbox(fitz.Rect(10,10,95,220),
        content_html('',16,0,False,'left',1.2,entry.html_body),css=css,archive=archive,scale_low=.7)
    assert spare>=0 and scale>.99
    path=tmp_path/'narrow.pdf';narrow.save(path);narrow.close()
    with fitz.open(path) as checked:
        result=checked[0]
        assert len(result.get_image_info())==1
        assert len(result.get_text('dict')['blocks'])>1
        assert '(' not in result.get_text() and '）' not in result.get_text()
        assert all(word in result.get_text() for word in ['Translated','prefix','translated','suffix'])
        image_box=fitz.Rect(result.get_image_info()[0]['bbox'])
        assert image_box.width==pytest.approx(expected_box.width,abs=.03)
        assert image_box.height==pytest.approx(expected_box.height,abs=.03)
        result.get_pixmap(matrix=fitz.Matrix(2,2)).save(tmp_path/'narrow.png')
    pdf.close()


def test_atomic_native_formula_does_not_capture_unowned_bracket(tmp_path):
    pdf,page,source,region,lines,images=original_inline_math(tmp_path)
    bracket=next(c for line in lines for c in line['chars'] if c['c']=='(')
    region.erase=[box for box in region.erase if box!=bracket['bbox']]
    assert native_parenthesized_asset_box(region,lines,images) is None
    pdf.close()


def test_atomic_native_formula_does_not_capture_intervening_prose(tmp_path):
    pdf,page,source,region,lines,images=original_inline_math(tmp_path)
    # A foreign glyph in the crop cannot be hidden merely because the source
    # descriptor omitted it. Its center lies between x and s on the same row.
    x_end=images[0]['bbox'][2]
    foreign={'c':'q','bbox':[x_end-.2,30,x_end+.2,40],
             'origin':[x_end-.2,45],'size':16,'font':'Helvetica'}
    lines=lines+[{'chars':[foreign]}]
    assert native_parenthesized_asset_box(region,lines,images) is None
    pdf.close()


def test_atomic_original_parentheses_are_limited_to_native_source(tmp_path):
    pdf,page,source,region,lines,images=original_inline_math(tmp_path)
    region.native=False
    assert native_parenthesized_asset_box(region,lines,images) is None
    region.native=True
    region.source='Input ⟦P000⟧⟦P001⟧ output'
    assert native_parenthesized_asset_box(region,lines,images) is None
    pdf.close()
