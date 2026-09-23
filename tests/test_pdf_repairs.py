import io
import pytest
import numpy as np
import pymupdf as fitz
from PIL import Image,ImageDraw,ImageFont
from slidetwin.extract import enrich_page,immutable_math_label
from slidetwin.models import Region
from slidetwin.qa import edge_rounding_pixel_count
from slidetwin.render import raster_label_patch,content_html,css_fonts
from slidetwin.config import Layout


def test_native_redaction_preserves_neighboring_math_with_overlapping_font_boxes(tmp_path):
    from slidetwin.models import Document,Page,digest
    from slidetwin.render import build_plan,render
    from slidetwin.qa import verify
    fitz.TOOLS.set_small_glyph_heights(True);source=tmp_path/'nearby.pdf'
    with fitz.open() as pdf:
        p=pdf.new_page(width=200,height=120)
        p.insert_text((30,50),'Label',fontsize=14)
        p.insert_text((55,68),'(-1)',fontsize=20)
        pdf.save(source);regions,_=enrich_page(p,1,[])
        document=Document(digest(source.read_bytes()),[Page(1,200,120,regions)])
    plan,failures=build_plan(source,document,{r.id:'标签' for r in regions},[1],Layout(),tmp_path)
    assert not failures
    output=tmp_path/'translated.pdf';render(source,output,plan,[1],Layout())
    with fitz.open(output) as pdf:assert '(-1)' in pdf[1].get_text()
    assert verify(source,output,[1],plan,tmp_path)['passed']


def test_native_label_also_printed_in_underlying_bitmap_is_removed(tmp_path):
    from slidetwin.models import Document,Page,digest
    from slidetwin.render import build_plan,render
    source=tmp_path/'duplicate.pdf';fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as bitmap_pdf:
        p=bitmap_pdf.new_page(width=240,height=120);p.insert_text((30,60),'GATE',fontsize=22,color=(0,.5,.2))
        raster=p.get_pixmap(matrix=fitz.Matrix(3,3)).tobytes('png')
    with fitz.open() as pdf:
        p=pdf.new_page(width=240,height=120);p.insert_image(p.rect,stream=raster);p.insert_text((30,60),'GATE',fontsize=22,color=(0,.5,.2))
        regions,_=enrich_page(p,1,[]);pdf.save(source)
    document=Document(digest(source.read_bytes()),[Page(1,240,120,regions)])
    plan,failures=build_plan(source,document,{regions[0].id:'门'},[1],Layout(),tmp_path);assert not failures
    output=tmp_path/'out.pdf';render(source,output,plan,[1],Layout())
    with fitz.open(output) as pdf:
        pix=pdf[1].get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(64,43,86,61),alpha=False)
        assert np.frombuffer(pix.samples,np.uint8).min()>245


def test_raster_multiline_font_height_is_measured_from_ink_not_paragraph_box():
    from slidetwin.render import raster_typography
    image=Image.new('RGB',(800,180),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    draw.text((15,20),'Continuous variables have fractional values.',font=font,fill=(0,70,180))
    draw.text((15,55),'This is a second line.',font=font,fill=(0,70,180))
    data=io.BytesIO();image.save(data,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=400,height=90);p.insert_image(p.rect,stream=data.getvalue())
        region=Region('r',1,'Continuous variables have fractional values. This is a second line.',[7,10,320,44],native=False,size=26)
        measured,rotate,_=raster_typography(p,region)
        assert 10<=measured.size<=16 and rotate==0
        assert measured.color & 255 > 140 and (measured.color>>16)<40


def test_shadow_copy_and_short_ocr_box_do_not_duplicate_visible_label():
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        p=pdf.new_page(width=300,height=150)
        p.insert_text((31,51),'Source',fontsize=20,color=(.75,.75,.75))
        p.insert_text((30,50),'Source',fontsize=20,color=(0,0,.4))
        p.insert_text((180,50),'Source',fontsize=20,color=(0,0,.4))
        # Both a native descriptor and a shorter OCR rediscovery exist.
        ds=[{'role':'text','ref':'native','text':'Source','bbox':fitz.Rect(29,30,95,55)},
            {'role':'text','ref':'ocr','text':'Source','bbox':fitz.Rect(29,44,95,51)}]
        regions,issues=enrich_page(p,1,ds)
        assert [r.source for r in regions]==['Source','Source']
        assert all(r.native for r in regions)
        assert len(regions[0].erase)==12
        assert any(x['kind']=='duplicate_native_ocr' for x in issues)


def test_metadata_author_is_preserved_while_heading_is_still_translated():
    with fitz.open() as pdf:
        pdf.set_metadata({'author':'Zhiping Lin'})
        p=pdf.new_page();p.insert_text((20,40),'Learning Pipeline');p.insert_text((20,80),'Zhiping Lin')
        regions,_=enrich_page(p,1,[])
        assert [r.source for r in regions]==['Learning Pipeline']


def test_timing_variable_labels_are_not_confused_with_english_instructions():
    for value in ('tsu-min','th-min','-min','Tclock','din','→dout','5ns ≤ tinv≤ 15ns','Tctk = 100ns'):
        assert immutable_math_label(value),value
    for value in ('Clock 5ns','Time 5ns','Minimum setup time','su marg'):
        assert not immutable_math_label(value),value
    for value in ('tcycle = T','Tjitter','tcs','er','ec','s = (⟦P002⟧−⟦P003⟧)/(⟦P004⟧−⟦P005⟧) q = clip(round(x/s), ⟦P000⟧, ⟦P001⟧)'):
        assert immutable_math_label(value),value


def test_natural_language_label_inside_formula_is_translated_separately():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'Symmetry: d(x,y) = d(y,x)',fontsize=14)
        descriptors=[{'role':'formula','ref':'formula','text':'Symmetry: d(x,y) = d(y,x)','bbox':fitz.Rect(18,30,260,60)}]
        regions,_=enrich_page(p,1,descriptors)
        assert [r.source for r in regions]==['Symmetry:']


def test_same_color_overprint_is_one_semantic_word():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'CIRCUIT',fontsize=20,color=(0,.5,.2));p.insert_text((20.1,50),'CIRCUIT',fontsize=20,color=(0,.5,.2))
        ds=[{'role':'text','ref':'r','text':'CIRCUIT','bbox':fitz.Rect(18,25,160,60)}]
        regions,_=enrich_page(p,1,ds)
        assert [r.source for r in regions]==['CIRCUIT']


def test_raster_cell_groups_wrapped_sentence_without_crossing_neighbor():
    from slidetwin.raster import raster_container
    bitmap=Image.new('RGB',(800,400),'white');draw=ImageDraw.Draw(bitmap)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    draw.rectangle((60,60,350,220),fill=(205,215,205),outline=(60,90,60),width=3)
    draw.rectangle((390,60,720,220),fill=(205,215,205),outline=(60,90,60),width=3)
    descriptors=[]
    for x,y,text in [(90,90,'temperature in'),(90,130,'Celsius'),(430,100,'Neighbor')]:
        draw.text((x,y),text,font=font,fill='red');bbox=draw.textbbox((x,y),text,font=font)
        descriptors.append({'role':'text','ref':text,'text':text,'bbox':fitz.Rect([v/2 for v in bbox])})
    data=io.BytesIO();bitmap.save(data,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=400,height=200);p.insert_image(p.rect,stream=data.getvalue())
        regions,_=enrich_page(p,1,descriptors)
        assert sorted(r.source for r in regions)==['Neighbor','temperature in Celsius']
        for r in regions:
            cell=raster_container(p,fitz.Rect(r.bbox));assert cell is not None
            assert cell.x1<190 if 'temperature' in r.source else cell.x0>190


def test_original_page_gate_only_allows_one_level_at_outermost_edge():
    left=np.zeros((20,20,3),np.uint8);right=left.copy();right[-1,:,0]=1
    assert edge_rounding_pixel_count(left,right)==20
    right[10,10,0]=1
    assert edge_rounding_pixel_count(left,right) is None
    right=left.copy();right[-1,10,0]=2
    assert edge_rounding_pixel_count(left,right) is None


def test_white_raster_label_cleanup_is_transparent_off_ink_and_keeps_adjacent_rule(tmp_path):
    bitmap=Image.new('RGB',(500,200),(20,80,160));draw=ImageDraw.Draw(bitmap)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',40)
    draw.text((120,60),'Gate',font=font,fill='white')
    bounds=draw.textbbox((120,60),'Gate',font=font)
    # A long nearby rule crosses the padded crop; it must not be inpainted.
    draw.line((0,bounds[3]+8,500,bounds[3]+8),fill='white',width=2)
    data=io.BytesIO();bitmap.save(data,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=250,height=100);p.insert_image(p.rect,stream=data.getvalue())
        r=Region('r',1,'Gate',[v/2 for v in bounds],native=False,size=16)
        path,box,color=raster_label_patch(p,r,tmp_path)
    patch=np.array(Image.open(path));assert color==0xffffff
    assert (patch[:,:,3]==0).mean()>.4
    row=round((bounds[3]/2+4-box[1])*4)
    assert not patch[row,:,3].any()


@pytest.mark.parametrize('size,image_height,baseline_down,width,height',[
    (18.025,20.275,6.756,123.66,21.68),
    (19.8,20.411,3.657,387.54,21.961),
])
def test_inline_subscript_image_reserves_adjacent_cjk_descender_space(tmp_path,size,image_height,baseline_down,width,height):
    fitz.TOOLS.set_small_glyph_heights(True)
    css,archive=css_fonts(Layout().fonts());im=Image.new('RGBA',(110,102),(0,0,0,255));data=io.BytesIO();im.save(data,format='PNG');archive.add((data.getvalue(),'m.png'))
    body=f'给出 <img src="m.png" style="width:21.97pt;height:{image_height}pt;vertical-align:-{baseline_down}pt;"> 的输出'
    frame=fitz.Rect(20,20,20+width,20+height)
    with fitz.open() as pdf:
        p=pdf.new_page();spare,scale=p.insert_htmlbox(frame,content_html('',size,0,True,'left',1.12,body),css=css,archive=archive,scale_low=.75)
        assert spare>=0 and scale>=.75
        assert all(fitz.Rect(w[:4]) in frame+(-.75,-.75,.75,.75) for w in p.get_text('words'))


def test_cell_reading_order_keeps_subscripts_with_their_own_baseline():
    from slidetwin.extract import ordered_cell_chars, protect_native_math
    def ch(c,x,y,size):
        return {'c':c,'origin':(x,y),'size':size,'bbox':(x,y-size,x+size*.5,y),
                'font':'Arial','flags':0,'color':0}
    glyphs=[ch('Q',10,40,18),ch('B',19,44.3,12),ch('0',25,44.3,12),ch(' ',31,40,18),
            *[ch(c,40+i*10,40,18) for i,c in enumerate('= bulk')],
            *[ch(c,10+i*10,60,18) for i,c in enumerate('voltage')]]
    ordered=ordered_cell_chars(glyphs)
    text,protected,_=protect_native_math(ordered,'')
    assert list(protected.values())==['QB0']
    assert 'bulk voltage' in text


def test_heading_reads_visual_top_line_before_larger_title():
    with fitz.open() as pdf:
        p=pdf.new_page()
        p.insert_text((20,100),'Sparsity',fontsize=25)
        p.insert_text((20,70),'PART 2.5: SPARSITY',fontsize=12)
        regions,_=enrich_page(p,1,[{'role':'section_header','ref':'title','text':'Sparsity PART 2.5: SPARSITY','bbox':fitz.Rect(18,50,200,110)}])
        assert regions[0].source.startswith('PART ')
        assert regions[1].source=='Sparsity'
        assert regions[0].size==12 and regions[1].size==25


def test_worked_arithmetic_misclassified_as_code_is_translated():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'Stored values: 16 x 9 = 144 values',fontsize=14)
        # Exercise native multiplication glyph independent of PDF encoding.
        from slidetwin import extract as ex
        original=ex.native_lines
        def multiplication(page):
            lines=original(page)
            for line in lines:
                for c in line['chars']:
                    if c['c']=='x':c['c']='×'
            return lines
        from unittest.mock import patch
        with patch.object(ex,'native_lines',multiplication):
            regions,_=enrich_page(p,1,[{'role':'code','ref':'r','text':'Stored values','bbox':fitz.Rect(18,30,300,60)}])
        assert len(regions)==1 and 'Stored values' in regions[0].source
        assert immutable_math_label('t setup')


def test_centered_paragraph_final_word_keeps_complete_translation_unit():
    with fitz.open() as pdf:
        p=pdf.new_page(width=600,height=150)
        text='Compression is optional: use it when the dense model misses storage, energy, throughput, or latency'
        p.insert_text((25,50),text,fontsize=11)
        center=25+fitz.get_text_length(text,fontsize=11)/2
        p.insert_text((center-fitz.get_text_length('targets.',fontsize=11)/2,63),'targets.',fontsize=11)
        regions,_=enrich_page(p,1,[])
        assert len(regions)==1
        assert regions[0].source.endswith('latency targets.')


def test_split_native_formula_suffix_stays_original_without_translated_neighbor():
    from slidetwin.render import fixed_native_suffix
    r=Region('r',1,'Given measurements {⟦P000⟧',[10,20,180,40],size=12,
             protected={'⟦P000⟧':'xraw'},protected_assets={'⟦P000⟧':{'bbox':[140,24,170,38],'baseline_down':2}},
             erase=[[10,20,20,32],[135,24,139,38],[140,24,145,36],[145,24,170,30]])
    lines=[{'chars':[{'c':'{','bbox':[135,24,139,38],'size':12}, {'c':'i','bbox':[145,35,149,42],'size':8}]}]
    result,suffix=fixed_native_suffix(r,lines)
    assert suffix=='{⟦P000⟧'
    assert result.erase==[[10,20,20,32]]


def test_parallel_labels_inside_one_box_do_not_merge_across_large_gap():
    with fitz.open() as pdf:
        p=pdf.new_page();p.draw_rect(fitz.Rect(10,25,280,65),color=(0,0,0))
        p.insert_text((20,50),'p-type',fontsize=14);p.insert_text((200,50),'n-type',fontsize=14)
        regions,_=enrich_page(p,1,[])
        assert [r.source for r in regions]==['p-type','n-type']


def test_formula_operating_modes_stay_on_their_original_rows():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((200,100),'cutoff',fontsize=20);p.insert_text((200,140),'linear',fontsize=20)
        regions,_=enrich_page(p,1,[{'role':'formula','ref':'f','text':'cutoff linear','bbox':fitz.Rect(190,70,300,150)}])
        assert [r.source for r in regions]==['cutoff','linear']
        assert abs(regions[1].bbox[1]-regions[0].bbox[1]-40)<.1


def test_split_alphabetic_word_recovers_exported_last_letter():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((100,100),'saturatio',fontsize=20)
        x=100+fitz.get_text_length('saturatio',fontsize=20)
        p.insert_text((x,100),'n',fontsize=20)
        regions,_=enrich_page(p,1,[{'role':'formula','ref':'f','text':'saturatio','bbox':fitz.Rect(95,70,x-.1,105)}])
        assert [r.source for r in regions]==['saturation']


def test_formula_subscript_and_chemical_fragment_remain_original():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'channel',fontsize=10)
        regions,_=enrich_page(p,1,[{'role':'formula','ref':'f','text':'channel','bbox':fitz.Rect(18,35,100,55)}])
        assert not regions
        assert immutable_math_label('SiO') and immutable_math_label('Vref')


def test_year_prose_beside_equation_is_not_lost_after_protection():
    from slidetwin.extract import needs_translation,protect
    source="In 1980's, VDD = 5V"
    assert needs_translation(source) and needs_translation(protect(source)[0])


def test_caption_badge_boundary_ignores_its_attached_thin_rule():
    from slidetwin.raster_refine import caption_body_box
    bitmap=Image.new('RGB',(600,100),'white');draw=ImageDraw.Draw(bitmap)
    draw.rectangle((10,10,150,60),fill=(0,110,150));draw.line((10,10,580,10),fill=(0,110,150),width=1)
    draw.rectangle((122,12,126,58),fill='white')
    data=io.BytesIO();bitmap.save(data,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=300,height=50);p.insert_image(p.rect,stream=data.getvalue())
        body=caption_body_box(p,fitz.Rect(4,4,292,35))
        assert body and 75<body.x0<80


def test_repeated_case_labels_reject_detector_box_covering_adjacent_circuit(tmp_path,monkeypatch):
    from slidetwin import raster_refine
    rows=[{'text':'Case1','score':.99,'bbox':[200,50,360,100]},
          {'text':'Case 2','score':.99,'bbox':[201,120,271,140]},
          {'text':'Case 3','score':.99,'bbox':[200,170,270,190]},
          {'text':'Case 4','score':.99,'bbox':[200,220,270,240]}]
    monkeypatch.setattr(raster_refine,'recognize',lambda *args:rows)
    with fitz.open() as pdf:
        page=pdf.new_page(width=500,height=300)
        result=raster_refine.refine(page,[{'text':'Case 2','bbox':fitz.Rect(200,120,271,140),'role':'text','ref':'ocr'}],tmp_path)
    assert len(result)==4 and result[0]['text']=='Case 1'
    assert result[0]['bbox'].width==70 and result[0]['bbox'].height==20


def test_subscript_misclassified_as_list_text_remains_original():
    with fitz.open() as pdf:
        p=pdf.new_page();p.insert_text((20,50),'Q',fontsize=24)
        p.insert_text((39,56),'channel',fontsize=14)
        p.insert_text((150,100),'channel',fontsize=14)
        regions,_=enrich_page(p,1,[{'role':'list_item','ref':'wrong-role','text':'Qchannel','bbox':fitz.Rect(18,25,110,65)}])
        assert len(regions)==1 and regions[0].source=='channel'
        assert regions[0].bbox[0]==150


def test_raster_descender_at_physical_page_edge_is_erased(tmp_path):
    bitmap=Image.new('RGB',(160,80),'white');draw=ImageDraw.Draw(bitmap)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',48)
    draw.text((45,38),'g',font=font,fill='black')
    data=io.BytesIO();bitmap.save(data,format='PNG')
    with fitz.open() as pdf:
        p=pdf.new_page(width=80,height=40);p.insert_image(p.rect,stream=data.getvalue())
        region=Region('r',1,'g',[22,22,39,40],size=24,native=False)
        path,_,_=raster_label_patch(p,region,tmp_path)
    patch=np.array(Image.open(path))
    assert patch[-1,:,3].max()>0


def test_tightly_stacked_labels_fit_chinese_inside_original_row_pitch(tmp_path):
    from slidetwin.extract import native_lines
    from slidetwin.models import Document,Page,digest
    from slidetwin.render import build_plan
    source=tmp_path/'stack.pdf'
    fitz.TOOLS.set_small_glyph_heights(True)
    with fitz.open() as pdf:
        p=pdf.new_page(width=300,height=160)
        p.insert_text((40,70),'inversion region',fontsize=17.64,color=(0,0,1))
        p.insert_text((40,86.2),'depletion region',fontsize=17.64,color=(.4,.5,.4))
        lines=native_lines(p);regions=[]
        for i,line in enumerate(lines):
            boxes=[list(c['bbox']) for c in line['chars']];box=fitz.Rect(boxes[0])
            for b in boxes[1:]:box|=fitz.Rect(b)
            regions.append(Region(f'r{i}',1,'label',list(box),size=17.64,erase=boxes,color=line['chars'][0]['color']))
        pdf.save(source)
    doc=Document(digest(source.read_bytes()),[Page(1,300,160,regions)])
    plans,failures=build_plan(source,doc,{'r0':'反型区','r1':'耗尽区'},[1],Layout(),tmp_path/'plan')
    assert not failures
    assert all(p.scale>=.75 for p in plans)
    assert max(b[3] for b in plans[0].glyph_boxes)<min(b[1] for b in plans[1].glyph_boxes)
