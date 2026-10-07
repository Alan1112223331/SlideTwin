"""Publish retained translations, isolating layout failures to their text blocks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import os
import re
import shutil
import subprocess

import pymupdf as fitz

from .extract import restore
from .models import digest, read_cache, write_json
from .qa import merge_final_checks, render_previews, verify
from .render import LayoutError, Placement, build_plan, render


def retained_values(work, document, config):
    values={}
    sources={r.id:r.source for p in document.pages for r in p.regions}
    sources.update({f'{r.id}_s{i}':s['source'] for p in document.pages for r in p.regions for i,s in enumerate(r.inline_styles)})
    literals={r.id:restore(r,r.source) for p in document.pages for r in p.regions}
    literals.update({f'{r.id}_s{i}':restore(r,s['source']) for p in document.pages for r in p.regions for i,s in enumerate(r.inline_styles)})
    def same_literal(key,value):
        return value==literals[key] if '⟦P' in sources[key] else value is None or value==literals[key]
    expected={'source_sha256':document.source_sha256,'config_fingerprint':config.fingerprint()}
    for name in ['translation-candidates.json','translation-ledger.json']:
        obj=read_cache(work/name)
        if any(obj.get(k)!=v for k,v in expected.items()):continue
        if name=='translation-candidates.json':
            targets=obj.get('targets',{})
            if isinstance(targets,dict):
                values.update({k:v['text'] for k,v in targets.items() if isinstance(v,dict) and
                               k in sources and v.get('source')==sources[k] and
                               same_literal(k,v.get('source_literal')) and
                               isinstance(v.get('text'),str) and v['text'].strip()})
        else:
            target_sources=obj.get('target_sources',{})
            target_literals=obj.get('target_source_literals',{})
            if not isinstance(target_sources,dict):target_sources={}
            if not isinstance(target_literals,dict):target_literals={}
            for field in ('accepted_candidates','translations'):
                candidates=obj.get(field,{})
                if isinstance(candidates,dict):
                    values.update({k:v for k,v in candidates.items() if k in sources and
                                   target_sources.get(k)==sources[k] and same_literal(k,target_literals.get(k)) and isinstance(v,str) and v.strip()})
    return values


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
    for key in failed:
        issues.append({'page':number,'id':key,'kind':'source_pixels_retained',
                       'reason':'Replacement did not pass local layout; preserve source pixels and retain the complete model text in the issues report'})
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
            issues.append({'page':number,'id':key,'kind':'source_pixels_retained',
                           'reason':'Insertion failed; rolled back this replacement without erasing its source'})
    try:
        qa=verify(source,rendered,[number],plans,local,phase='intermediate')
        issues.extend(qa['failures'])
    except Exception as exc:
        issues.append({'page':number,'kind':'intermediate_verification_unavailable','type':type(exc).__name__})
    return rendered,failed,issues,plans


def publish_best_effort(source, output, work, document, selected, config, reason, values=None, preview=True, log=print,preflight=None):
    """Atomically publish every selected page; never claim warning-free QA."""
    if digest(source.read_bytes())!=document.source_sha256:
        raise RuntimeError('Source changed; cannot export retained translations')
    root=work/'best-effort';root.mkdir(parents=True,exist_ok=True)
    values={**retained_values(work,document,config),**(values or {})}
    doc=deepcopy(document)
    report={'status':'completed_with_warnings','reason':str(reason),'source_sha256':document.source_sha256,
            'config_fingerprint':config.fingerprint(),'selected_pages':selected,'pages':[],
            'fully_validated':False,'visual_review':'pending_human_review','diagnostics_location':'issues_report'}
    ledger=read_cache(work/'translation-ledger.json')
    if ledger.get('source_sha256')==document.source_sha256 and ledger.get('config_fingerprint')==config.fingerprint():
        report['translation_failures']=ledger.get('failures',[])
        report['preparation_warnings']=ledger.get('preparation_warnings',[])
    candidate=root/'all-pages.pdf'
    final_plans=[]
    with fitz.open(source) as original,fitz.open() as result:
        for number in selected:
            page=doc.pages[number-1];local=root/f'page-{number:04d}';local.mkdir(exist_ok=True)
            missing=[]
            for region in page.regions:
                if not isinstance(values.get(region.id),str) or not values[region.id].strip():
                    missing.append(region.id);values[region.id]='[未取得译文]'
                values[region.id]=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]','',values[region.id])
            result.insert_pdf(original,from_page=number-1,to_page=number-1)
            rendered,failed,issues,plans=render_local(source,doc,values,number,config.layout,local,preflight)
            final_plans.extend(plans)
            extraction_failed=not page.regions and any(i.get('blocking') for i in issues)
            with fitz.open(rendered) as translated:
                # Production slides keep their source geometry. Diagnostics and
                # unplaced model text belong in the sidecar, never on the slide.
                result.insert_pdf(translated,from_page=1,to_page=1)
            unplaced=[{'id':r.id,'bbox':r.bbox,
                       'translation':restore(r,values[r.id]) if r.id not in missing else None}
                      for r in page.regions if r.id in failed]
            report['pages'].append({'source_page':number,'layout':'source_layout','missing_targets':missing,
                                    'overflow_targets':sorted(failed),'unplaced_translations':unplaced,
                                    'extraction_failed':extraction_failed,'issues':issues})
        result.set_metadata({'title':source.stem+' - SlideTwin','producer':'SlideTwin'})
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
    try:
        write_json(work/'final-layout-plan.json',{'placements':[asdict(p) for p in final_plans],
                   'selected_pages':selected,'output_sha256':report['output_sha256']})
    except Exception as exc:
        report.setdefault('diagnostic_warnings',[]).append({'kind':'final_plan_write_failed','type':type(exc).__name__})
        log(f'PDF published; final placement evidence could not be saved ({type(exc).__name__})')
    try:
        report=merge_final_checks(report,verify(source,output,selected,final_plans,work/'final-qa',phase='final'))
    except Exception as exc:
        report.setdefault('diagnostic_warnings',[]).append({'kind':'final_verification_unavailable','type':type(exc).__name__})
        log(f'PDF published; final verification could not complete ({type(exc).__name__})')
    for path,value in [(root/'used-translations.json',values),(work/'run.json',report),(output.with_suffix('.issues.json'),report)]:
        try:write_json(path,value)
        except Exception as exc:
            report.setdefault('diagnostic_warnings',[]).append({'kind':'diagnostic_write_failed','type':type(exc).__name__})
            log(f'PDF published; a diagnostic file could not be saved ({type(exc).__name__})')
    if preview:
        try:report['preview']=render_previews(output,root,selected,config.layout.render_dpi)
        except (RuntimeError,OSError,subprocess.SubprocessError) as exc:report['preview_error']=str(exc)
    for path in [work/'run.json',output.with_suffix('.issues.json')]:
        try:write_json(path,report)
        except Exception:pass  # Already reported above; never hide the published PDF.
    log(f'Created {output} ({len(selected)*2} pages, WITH WARNINGS). Diagnostics and unplaced model text: {output.with_suffix(".issues.json")}')
    return report
