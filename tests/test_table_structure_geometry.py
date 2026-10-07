"""A bordered cell retains measured row structure without splitting meaning."""
import pymupdf as fitz
from slidetwin.extract import enrich_page, table_physical_line_styles


def test_bordered_native_cell_keeps_three_original_rows_as_style_correspondences():
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        page=pdf.new_page(width=350,height=170)
        page.draw_rect(fitz.Rect(20,28,180,74),color=(0,0,0))
        texts=['Dichotomous','Two-category','Two-class']
        descriptors=[]
        for index,text in enumerate(texts):
            baseline=41+index*14
            page.insert_text((35,baseline),text,fontsize=12)
            descriptors.append({'ref':text,'role':'text','text':text,'bbox':fitz.Rect(33,baseline-12,150,baseline+3)})
        regions,_=enrich_page(page,1,descriptors)
    assert len(regions)==1 and regions[0].role=='table_cell'
    assert regions[0].source=='Dichotomous Two-category Two-class'
    styles=regions[0].inline_styles
    assert [style['source'] for style in styles]==texts
    assert [style['source_line_index'] for style in styles]==[0,1,2]
    assert [style['break_before'] for style in styles]==[False,True,True]
    assert not any(style['break_after'] for style in styles)


def test_same_source_style_index_is_preserved_and_not_duplicated():
    def glyph(c,x,y):
        return {'c':c,'origin':(x,y),'bbox':(x,y-10,x+5,y+2),
                'font':'Arial','size':12,'color':0,'flags':0}
    chars=[*[glyph(c,20+i*5,40) for i,c in enumerate('First row')],
           *[glyph(c,20+i*5,58) for i,c in enumerate('Second row')]]
    original=[{'source':'Second row','color':0xFF0000,'bold':True,'italic':False}]
    styles=table_physical_line_styles(chars,original,{})
    assert len(styles)==2
    assert styles[0]['source']=='Second row' and styles[0]['color']==0xFF0000 and styles[0]['bold']
    assert styles[0]['break_before'] and styles[0]['source_line_index']==1
    assert styles[1]['source']=='First row' and not styles[1]['break_before']
    assert original==[{'source':'Second row','color':0xFF0000,'bold':True,'italic':False}]


def test_synthetic_paragraph_merge_does_not_invent_table_break_targets():
    with fitz.open() as pdf:
        page=pdf.new_page(width=600,height=150)
        text='Compression is optional: use it when the dense model misses storage, energy, throughput, or latency'
        page.insert_text((25,50),text,fontsize=11)
        center=25+fitz.get_text_length(text,fontsize=11)/2
        page.insert_text((center-fitz.get_text_length('targets.',fontsize=11)/2,63),'targets.',fontsize=11)
        regions,_=enrich_page(page,1,[])
    assert len(regions)==1 and regions[0].source.endswith('latency targets.')
    assert not any(style.get('structural_line') for style in regions[0].inline_styles)


def test_stacked_script_stays_with_its_physical_table_row():
    def glyph(c,x,y,size=12):
        return {'c':c,'origin':(x,y),'bbox':(x,y-size*.8,x+size*.5,y+size*.2),
                'font':'Arial','size':size,'color':0,'flags':0}
    chars=[glyph('x',20,40),glyph('3',26,36,8),glyph('1',26,44,8),
           *[glyph(c,20+i*6,60) for i,c in enumerate('Next row')]]
    styles=table_physical_line_styles(chars,[],{})
    assert [style['source'] for style in styles]==['x31','Next row']
    assert [style['break_before'] for style in styles]==[False,True]
