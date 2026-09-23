"""Regression for a failed incremental plan accidentally becoming the baseline."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slidetwin.models import digest


def test_failed_refresh_preserves_the_published_plan_and_pdf(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import refresh_verified_output as module
    monkeypatch.setattr(module,'B',tmp_path)
    work=tmp_path/'course'/'async';work.mkdir(parents=True)
    out=tmp_path/'pdf'/'TEST-bilingual.pdf';out.parent.mkdir();out.write_bytes(b'prior verified PDF')
    case={'course':'TEST','pages':1,'work':str(work.parent),'input':str(tmp_path/'source.pdf')}
    (tmp_path/'selection.json').write_text(json.dumps({'cases':[case]}))
    old=json.dumps({'placements':[{'page':1,'text':'published'}]}).encode()
    (work/'layout-plan.json').write_bytes(old)
    (work/'finish.json').write_text(json.dumps({'status':'automated_checks_passed','output_sha256':digest(out.read_bytes())}))
    for name in ['qa.json','final-translation-ledger.json']:(work/name).write_text('{}')
    monkeypatch.setattr(module,'load_config',lambda:SimpleNamespace(layout=None))
    monkeypatch.setattr(module,'verified_pages',lambda *args:(None,{'r':'translation'},[1]))
    def fail(*args):
        (work/'layout-plan.json').write_text('{"placements":[],"failures":["overlap"]}')
        return [],['overlap']
    monkeypatch.setattr(module,'build_plan',fail)
    with pytest.raises(AssertionError,match='overlap'):module.main('TEST')
    assert (work/'layout-plan.json').read_bytes()==old
    assert out.read_bytes()==b'prior verified PDF'
