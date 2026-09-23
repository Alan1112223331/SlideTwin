"""Verify cached translations and publish each COMPLETE deck after layout QA."""
import argparse,json,os,shutil,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from slidetwin.config import Settings,Provider,Translation,Layout
from slidetwin.models import Document,digest,write_json
from slidetwin.async_translate import AsyncTranslator,ASYNC_VERSION
from slidetwin.protocol import parse_content_response as parse_response,ProtocolError
from slidetwin.render import build_plan,render
from slidetwin.qa import verify,render_previews

ROOT=Path(__file__).resolve().parents[1];B=ROOT/'output/four-course-20260921'
def load_config():
    s=json.loads((B/'async-settings.json').read_text(encoding='utf-8'))
    return Settings(Provider(**s['provider']),Translation(**s['translation']),Layout(**s['layout']))
def verified_pages(case,config):
    source=Path(case['input']);work=Path(case['work'])/'async';doc=Document.load(work/'document.json')
    t=AsyncTranslator(config,None,doc,source,work,log=lambda _:None)
    t.glossary=json.loads((work/'translations/glossary.json').read_text(encoding='utf-8'))['text']
    values={};selected=[]
    for page in doc.pages:
        path=work/f'translations/page-{page.number:04d}.json'
        if not path.exists():continue
        obj=json.loads(path.read_text(encoding='utf-8'))
        key=digest(ASYNC_VERSION+config.fingerprint()+doc.source_sha256+t.context+t.glossary+json.dumps(asdict(page),sort_keys=True))
        if obj.get('key')!=key or not obj.get('reviewed'):continue
        sources=t.sources(page)
        try:
            checked=parse_response(json.dumps(obj['translations']),sources,'json')
        except ProtocolError:continue
        values.update(checked);selected.append(page.number)
    return doc,values,selected
def finish(case,config,partial=False):
    doc,values,selected=verified_pages(case,config)
    if not selected or (not partial and len(selected)!=case['pages']):return None
    course=case['course'];work=Path(case['work'])/'async';source=Path(case['input'])
    target=B/'diagnostics/async-layout'/course if partial else work
    target.mkdir(parents=True,exist_ok=True)
    print(datetime.now().isoformat(timespec='seconds'),course,'Layout',len(selected),'/',case['pages'],flush=True)
    plans,failures=build_plan(source,doc,values,selected,config.layout,target)
    report={'course':course,'selected_pages':selected,'complete_deck':len(selected)==case['pages'],'layout_failures':failures}
    if failures:
        report['status']='layout_blocked';write_json(target/'finish.json',report)
        print(course,'BLOCKED',len(failures),flush=True);return report
    candidate=target/'candidate.pdf';render(source,candidate,plans,selected,config.layout)
    write_json(target/'final-translation-ledger.json',{'source_sha256':doc.source_sha256,'config_fingerprint':config.fingerprint(),
               'selected_pages':selected,'translations':values,'note':'Read from verified final page caches, including separately audited semantic follow-ups.'})
    qa=verify(source,candidate,selected,plans,target);report['qa']=qa
    if not qa['passed']:
        report['status']='qa_blocked';write_json(target/'finish.json',report);print(course,'QA FAILED',qa['failures'][:5],flush=True);return report
    report['previews']=render_previews(candidate,target,selected,config.layout.render_dpi)
    if digest(source.read_bytes())!=doc.source_sha256:raise RuntimeError('Source changed')
    if not partial:
        out=B/'pdf'/f'{course}-bilingual.pdf';out.parent.mkdir(exist_ok=True)
        tmp=out.with_suffix('.pdf.tmp');shutil.copy2(candidate,tmp);os.replace(tmp,out)
        report.update(output=str(out),output_sha256=digest(out.read_bytes()),status='automated_checks_passed',visual_review='pending')
    else:report['status']='partial_diagnostic_only'
    report['finished_at']=datetime.now(timezone.utc).isoformat();write_json(target/'finish.json',report)
    print(course,report['status'],flush=True);return report
def main(args):
    manifest=json.loads((B/'selection.json').read_text(encoding='utf-8'));done=set()
    while True:
        config=load_config()
        for case in manifest['cases']:
            if args.course and case['course']!=args.course:continue
            if case['course'] in done:continue
            result=finish(case,config,args.partial)
            if result is not None:done.add(case['course'])
        if not args.watch or len(done)==len([c for c in manifest['cases'] if not args.course or c['course']==args.course]):break
        time.sleep(10)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--partial',action='store_true');p.add_argument('--course');main(p.parse_args())
