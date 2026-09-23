"""Re-render changed plans, then verify the complete reassembled deck."""
import argparse,json,os,shutil
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import pymupdf as fitz
from finish_async_four_courses import B,load_config,verified_pages
from slidetwin.models import digest,write_json
from slidetwin.render import build_plan,render
from slidetwin.qa import verify,render_previews

def main(course):
    case=next(c for c in json.loads((B/'selection.json').read_text(encoding='utf-8'))['cases'] if c['course']==course)
    cfg=load_config();doc,values,pages=verified_pages(case,cfg)
    assert len(pages)==case['pages'],'Incomplete reviewed translation'
    work=Path(case['work'])/'async';source=Path(case['input']);out=B/'pdf'/f'{course}-bilingual.pdf'
    prior=digest(out.read_bytes());finish=json.loads((work/'finish.json').read_text(encoding='utf-8'))
    assert finish['status']=='automated_checks_passed' and finish['output_sha256']==prior
    old_plan=(work/'layout-plan.json').read_bytes()
    if finish.get('layout_plan_sha256'):
        assert digest(old_plan)==finish['layout_plan_sha256'],'Published plan changed outside refresh'
    old=json.loads(old_plan)['placements']
    audit=work/'output-revisions'/datetime.now().strftime('%Y%m%d-%H%M%S');audit.mkdir(parents=True)
    for name in ['layout-plan.json','final-translation-ledger.json','finish.json','qa.json']:
        shutil.copy2(work/name,audit/name)
    print(course,'Building current plans',flush=True)
    try:
        plans,failures=build_plan(source,doc,values,pages,cfg.layout,work)
        new_plan=(work/'layout-plan.json').read_bytes()
    finally:
        # A failed staging run must not become the comparison baseline for the
        # next refresh. The baseline describes the currently published PDF.
        (work/'layout-plan.json').write_bytes(old_plan)
    assert not failures,failures
    changed=[n for n in pages if [p for p in old if p['page']==n]!=[asdict(p) for p in plans if p.page==n]]
    renderer=digest((Path(__file__).resolve().parents[1]/'src/slidetwin/render.py').read_bytes())
    if finish.get('render_source_sha256') and renderer!=finish['render_source_sha256']:
        changed=list(pages)
    print(course,'Changed pages',changed,flush=True)
    candidate=audit/'candidate.pdf'
    if changed:
        patch=audit/'changed-pages.pdf';render(source,patch,plans,changed,cfg.layout)
        with fitz.open(out) as prev,fitz.open(patch) as updated,fitz.open() as result:
            for i,n in enumerate(pages):
                pdf=updated if n in changed else prev
                j=changed.index(n)*2 if n in changed else i*2
                result.insert_pdf(pdf,from_page=j,to_page=j+1)
            result.save(candidate,garbage=4,deflate=True)
    else:shutil.copy2(out,candidate)
    qa=verify(source,candidate,pages,plans,work)
    assert qa['passed'],qa['failures']
    previews=render_previews(candidate,work,pages,cfg.layout.render_dpi)
    assert prior==digest(out.read_bytes()),'Output changed concurrently'
    assert verified_pages(case,load_config())[1]==values,'Translation changed concurrently'
    assert digest(source.read_bytes())==doc.source_sha256
    temp=out.with_suffix('.pdf.tmp');shutil.copy2(candidate,temp);os.replace(temp,out)
    write_json(work/'layout-plan.json',json.loads(new_plan))
    write_json(work/'final-translation-ledger.json',{'source_sha256':doc.source_sha256,'config_fingerprint':cfg.fingerprint(),'selected_pages':pages,'translations':values})
    finish.update(output_sha256=digest(out.read_bytes()),layout_plan_sha256=digest((work/'layout-plan.json').read_bytes()),render_source_sha256=renderer,qa=qa,previews=previews,visual_review='pending',finished_at=datetime.now(timezone.utc).isoformat(),revision={'changed_pages':changed,'prior_sha256':prior,'audit':str(audit)})
    write_json(work/'finish.json',finish)
    print(course,'automated_checks_passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--course',required=True);main(parser.parse_args().course)
