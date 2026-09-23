"""Publish retained translations, isolating layout failures to their text blocks."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from html import escape
import os
import re
import shutil
import subprocess

import pymupdf as fitz

from .extract import restore
from .models import digest, read_cache, write_json
from .qa import render_previews, verify
from .render import LayoutError, Placement, build_plan, render, css_fonts


def retained_values(work, document, config):
    values={}
    expected={'source_sha256':document.source_sha256,'config_fingerprint':config.fingerprint()}
    for name in ['translation-candidates.json','translation-ledger.json']:
        obj=read_cache(work/name)
        if any(obj.get(k)!=v for k,v in expected.items()):continue
        if name=='translation-candidates.json':
            targets=obj.get('targets',{})
            if isinstance(targets,dict):
                values.update({k:v['text'] for k,v in targets.items() if isinstance(v,dict) and isinstance(v.get('text'),str) and v['text'].strip()})
        else:
            for field in ('accepted_candidates','translations'):
                candidates=obj.get(field,{})
                if isinstance(candidates,dict):
                    values.update({k:v for k,v in candidates.items() if isinstance(v,str) and v.strip()})
    return values


def erase_only(region):
    return Placement(region.id,region.page,'',region.bbox,region.bbox,region.size,1,'left',region.bold,
                     region.color,True,region.erase)


def render_local(source,document,values,number,layout,local,preflight=None):
    """Retain all successful placements; quarantine only offending regions."""
    if preflight is None:
        plans,failures=build_plan(source,document,values,[number],layout,local)
    else:
        plans=[p for p in preflight[0] if p.page==number]
        failures=[f for f in preflight[1] if f.get('page')==number]
    regions={r.id:r for r in document.pages[number-1].regions}
    failed={f['id'] for f in failures if f.get('id') in regions}
    for failure in failures:
        if failure.get('kind')=='translated_text_overlap':failed.update(failure.get('ids',[]))
    issues=list(failures)
    plans=[p for p in plans if p.id not in failed]
    cleared=set()
    for key in failed:
        region=regions[key]
        if region.native and region.erase:
            plans.append(erase_only(region));cleared.add(key)
        else:
            issues.append({'id':key,'kind':'source_pixels_retained','reason':'No safe local erase geometry; translation is in the note below'})
    rendered=local/'rendered.pdf'
    while True:
        try:
            render(source,rendered,plans,[number],layout)
            break
        except LayoutError as exc:
            key=exc.region_id
            if key is None or not any(p.id==key for p in plans):raise
            plans=[p for p in plans if p.id!=key]
            failed.add(key);issues.append({'id':key,'kind':'local_render_failure','reason':str(exc)})
            region=regions[key]
            if key not in cleared and region.native and region.erase:
                plans.append(erase_only(region));cleared.add(key)
            else:
                issues.append({'id':key,'kind':'source_pixels_retained','reason':'Cannot safely remove this source label'})
    qa=verify(source,rendered,[number],plans,local)
    issues.extend(qa['failures'])
    return rendered,failed,issues


def append_local_notes(result,translated,page,failed,values,layout,extraction_failed=False):
    """Keep the slide at its original scale; put only overflow blocks below it."""
    css,archive=css_fonts(layout.fonts())
    paragraphs=[]
    if extraction_failed:
        paragraphs.append('<p>该页文字提取未完成，未取得可放置的译文。已保留原页图形，详情见 issues.json。</p>')
    for region in page.regions:
        if region.id not in failed:continue
        x,y=round(region.bbox[0]),round(region.bbox[1])
        paragraphs.append(f'<p><span style="color:#775521">{escape(region.id)}（原位置 {x}, {y}）</span><br>{escape(restore(region,values[region.id]))}</p>')
    body='<div style="font-family:twin,latin,unicode,extended,symbols;font-size:11pt;line-height:1.3"><b>局部排版待检查：以下译文超出原文字区域，原页图形和其它译文已保留。</b>'+''.join(paragraphs)+'</div>'
    width=max(page.width,320)
    height=120
    while True:
        with fitz.open() as probe:
            p=probe.new_page(width=width,height=height)
            spare,_=p.insert_htmlbox(fitz.Rect(16,10,width-16,height-10),body,css=css,archive=archive,scale_low=1)
        if spare>=0:break
        height*=1.5
    # Failed measurement attempts never touch completed output pages.
    target=result.new_page(width=width,height=page.height+height)
    target.show_pdf_page(fitz.Rect(0,0,page.width,page.height),translated,1)
    target.draw_rect(fitz.Rect(0,page.height,width,page.height+height),color=None,fill=(1,.98,.92))
    target.insert_htmlbox(fitz.Rect(16,page.height+10,width-16,page.height+height-10),body,css=css,archive=archive,scale_low=1)


def publish_best_effort(source, output, work, document, selected, config, reason, values=None, preview=True, log=print,preflight=None):
    """Atomically publish every selected page; never claim warning-free QA."""
    if digest(source.read_bytes())!=document.source_sha256:
        raise RuntimeError('Source changed; cannot export retained translations')
    root=work/'best-effort';root.mkdir(parents=True,exist_ok=True)
    values={**retained_values(work,document,config),**(values or {})}
    doc=deepcopy(document)
    report={'status':'completed_with_warnings','reason':str(reason),'source_sha256':document.source_sha256,
            'config_fingerprint':config.fingerprint(),'selected_pages':selected,'pages':[],
            'fully_validated':False,'visual_review':'pending_human_review'}
    ledger=read_cache(work/'translation-ledger.json')
    if ledger.get('source_sha256')==document.source_sha256 and ledger.get('config_fingerprint')==config.fingerprint():
        report['translation_failures']=ledger.get('failures',[])
        report['preparation_warnings']=ledger.get('preparation_warnings',[])
    candidate=root/'all-pages.pdf'
    with fitz.open(source) as original,fitz.open() as result:
        for number in selected:
            page=doc.pages[number-1];local=root/f'page-{number:04d}';local.mkdir(exist_ok=True)
            missing=[]
            for region in page.regions:
                if not isinstance(values.get(region.id),str) or not values[region.id].strip():
                    missing.append(region.id);values[region.id]='[未取得译文]'
                values[region.id]=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]','',values[region.id])
            result.insert_pdf(original,from_page=number-1,to_page=number-1)
            rendered,failed,issues=render_local(source,doc,values,number,config.layout,local,preflight)
            extraction_failed=not page.regions and any(i.get('blocking') for i in issues)
            mode='local_overflow_notes' if failed or extraction_failed else 'source_layout'
            with fitz.open(rendered) as translated:
                if failed or extraction_failed:append_local_notes(result,translated,page,failed,values,config.layout,extraction_failed)
                else:result.insert_pdf(translated,from_page=1,to_page=1)
            report['pages'].append({'source_page':number,'layout':mode,'missing_targets':missing,
                                    'overflow_targets':sorted(failed),'issues':issues})
        result.set_metadata({'title':source.stem+' - SlideTwin 待检查译稿','subject':'Retained model translations; see accompanying issues report','producer':'SlideTwin'})
        result.subset_fonts();result.save(candidate,garbage=4,deflate=True)
    with fitz.open(candidate) as check:
        if len(check)!=2*len(selected):raise RuntimeError('Best-effort export page count mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        backup=output.with_name(output.stem+'.previous-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+output.suffix)
        shutil.copy2(output,backup);report['previous_output_backup']=str(backup)
    if digest(source.read_bytes())!=document.source_sha256:raise RuntimeError('Source changed during export')
    temp=output.with_name(output.name+'.slidetwin.tmp')
    try:
        shutil.copy2(candidate,temp);os.replace(temp,output)
    finally:temp.unlink(missing_ok=True)
    report.update(output=str(output),output_sha256=digest(output.read_bytes()),output_pages=len(selected)*2,
                  model_text_targets=sum(1 for p in doc.pages if p.number in selected for r in p.regions if not values[r.id].startswith('[未取得译文')),
                  final_output_published=True)
    write_json(root/'used-translations.json',values)
    write_json(work/'run.json',report);write_json(output.with_suffix('.issues.json'),report)
    if preview:
        try:report['preview']=render_previews(output,root,selected,config.layout.render_dpi)
        except (RuntimeError,OSError,subprocess.SubprocessError) as exc:report['preview_error']=str(exc)
    write_json(work/'run.json',report);write_json(output.with_suffix('.issues.json'),report)
    log(f'Created {output} ({len(selected)*2} pages, WITH WARNINGS). Model text was retained; see {output.with_suffix(".issues.json")}')
    return report
