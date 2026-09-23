"""Recover prose hidden inside Docling's otherwise empty raster formula blocks."""
import re
import numpy as np
import pymupdf as fitz
from .models import digest,write_json

def supplement(page,descriptors,work):
    import json
    additions=[]
    anchors={'what','if','when','where','for','every','time','violation','delay','setup','hold','given','assume','therefore'}
    for descriptor in descriptors:
        if descriptor['role']!='formula' or len(page.get_textbox(descriptor['bbox']).strip())>=3:continue
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2),clip=descriptor['bbox'],alpha=False)
        key=digest(pix.samples+'formula-prose-v2'.encode());path=work/'formula-ocr'/f'{key}.json'
        if path.exists():rows=json.loads(path.read_text(encoding='utf-8'))['rows']
        else:
            import torch
            from rapidocr import RapidOCR,EngineType
            if not hasattr(supplement,'engine'):
                torch.set_num_threads(min(4,torch.get_num_threads()))
                supplement.engine=RapidOCR(params={'Global.log_level':'warning','Global.return_word_box':True,
                    'Det.engine_type':EngineType.TORCH,'Cls.engine_type':EngineType.TORCH,'Rec.engine_type':EngineType.TORCH})
            result=supplement.engine(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3])
            rows=[]
            for line_box,words in zip(result.boxes if result.boxes is not None else [],result.word_results):
                run=[]
                def flush():
                    if not run or not any(w[0].strip('.,:!?').lower() in anchors for w in run):return
                    points=np.array([point for _,_,box in run for point in box]);lb=np.array(line_box)
                    # CTC word boxes divide wide initial letters approximately.
                    # The detector's line edge contains the complete first glyph.
                    initial=run[0][0]==words[0][0] and np.array_equal(run[0][2],words[0][2])
                    left=float(np.array(line_box)[:,0].min()) if initial else float(points[:,0].min())
                    rows.append({'text':' '.join(w[0] for w in run),'bbox':[(pix.x+left)/2,
                        (pix.y+float(lb[:,1].min()))/2,(pix.x+float(points[:,0].max()))/2,(pix.y+float(lb[:,1].max()))/2]})
                for word,score,box in words:
                    if score>=.94 and re.fullmatch(r'[A-Za-z]{2,}[.,:!?]?',word) and not word.isupper():run.append((word,score,box))
                    else:flush();run=[]
                flush()
            write_json(path,{'key':key,'engine':'RapidOCR/PyTorch','rows':rows})
        for index,row in enumerate(rows):
            box=fitz.Rect(row['bbox'])
            if any(d['role']!='formula' and (box & d['bbox']).get_area()>.8*box.get_area() for d in descriptors):continue
            additions.append({'ref':descriptor['ref']+f'/prose/{index}','role':'text','text':row['text'],'bbox':box})
    return additions
