from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from datetime import datetime

import pymupdf as fitz
from filelock import FileLock, Timeout

from .client import ModelClient, ProviderError
from .protocol import ProtocolError
from .config import Settings
from .extract import extract
from .models import digest, write_json, read_cache
from .qa import render_previews, verify
from .render import LayoutError, build_plan, render
from .translate import Translator
from .model_pool import AsyncModelPool
from .async_translate import AsyncTranslator


def parse_pages(value: str | None, count: int) -> list[int]:
    if not value:
        return list(range(1, count+1))
    result = []
    for part in value.split(","):
        ends = part.strip().split("-")
        if len(ends) == 1:
            result.append(int(ends[0]))
        elif len(ends) == 2:
            start, end = map(int, ends)
            if end < start:
                raise ValueError("Page ranges must be increasing")
            result.extend(range(start, end+1))
        else:
            raise ValueError("Use pages like 1,3-5,8")
    if not result or len(set(result)) != len(result) or any(x < 1 or x > count for x in result):
        raise ValueError(f"Pages must be unique and within 1..{count}")
    return result


def run(source: Path, output: Path, work: Path, config: Settings, pages=None, preview=True, force_extract=False, log=print) -> dict:
    source, output, work = source.resolve(), output.resolve(), work.resolve()
    if source.suffix.lower() != ".pdf":
        raise ValueError("This release supports PDF course slides. PPTX must not be silently flattened; native PPTX editing is not yet implemented.")
    if not source.is_file() or source == output:
        raise ValueError("Input must exist and output must differ from input")
    config.validate()
    config.layout.fonts()
    config.provider.key()
    initial_hash = digest(source.read_bytes())
    work.mkdir(parents=True, exist_ok=True)
    candidate = work / "candidate.pdf"
    if source in {candidate, work/"selected-source.pdf"} or output in {candidate, work/"selected-source.pdf"}:
        raise ValueError("Input/output conflicts with internal working artifacts")
    with fitz.open(source) as pdf:
        selected = parse_pages(pages, len(pdf))
    try:
        with FileLock(work/"run.lock", timeout=0):
            write_json(work/"run.json", {"status": "running", "source_sha256": initial_hash, "selected_pages": selected})
            document = extract(source, work, selected, force=force_extract, log=log)
            import asyncio
            async def translate_async():
                async_client = AsyncModelPool(config.provider, trace_path=work/"request-timings.jsonl")
                try:
                    values = await AsyncTranslator(config, async_client, document, source, work, log=log).run_async(selected)
                    return values, dict(async_client.usage)
                finally:
                    await async_client.close()
            translations, usage = asyncio.run(translate_async())
            log("Layout: planning all placements before changing any PDF object")
            placements, failures = build_plan(source, document, translations, selected, config.layout, work)
            if failures and config.layout.strict:
                write_json(work/"run.json", {"status": "blocked", "stage": "layout", "failures": failures})
                raise LayoutError(f"{len(failures)} layout/extraction issues require local recovery; see layout-plan.json")
            render(source, candidate, placements, selected, config.layout)
            report = verify(source, candidate, selected, placements, work)
            report["layout_failures"] = failures
            report["usage"] = usage
            report['preparation_warnings']=read_cache(work/'translation-ledger.json').get('preparation_warnings',[])
            if failures:
                report["passed"] = False
            if digest(source.read_bytes()) != initial_hash:
                raise RuntimeError("Source file changed during the run; output will not be published")
            report["source_unchanged"] = True
            if not report["passed"]:
                report["status"] = "needs_review"
                write_json(work/"qa.json", report)
                write_json(work/"run.json", report)
                raise LayoutError("Quality checks need review; retaining the candidate and exporting with local recovery")
            output.parent.mkdir(parents=True, exist_ok=True)
            # Copy to the destination filesystem and atomically replace only after
            # all checks pass. Keeps a previous good result intact on any failure.
            temp = output.with_name(output.name + ".slidetwin.tmp")
            try:
                if output.exists():
                    backup=output.with_name(output.stem+'.previous-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+output.suffix)
                    shutil.copy2(output,backup);report['previous_output_backup']=str(backup)
                shutil.copyfile(candidate, temp)
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)
            report["output"] = str(output)
            report["status"] = "automated_checks_passed"
            report['final_output_published']=True
            if preview:
                try:report['preview']=render_previews(output,work,selected,config.layout.render_dpi)
                except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
                    report['preview_error']=str(exc)
                    log(f'PDF published; preview generation failed: {exc}')
            if report.get('preparation_warnings') or report.get('preview_error'):
                report['status']='completed_with_warnings'
                write_json(output.with_suffix('.issues.json'),report)
            elif output.with_suffix('.issues.json').exists():
                # A recovered run must not leave the previous failure report
                # apparently attached to its new PDF.
                write_json(output.with_suffix('.issues.json'),report)
            write_json(work/"qa.json", report)
            write_json(work/"run.json", report)
            log(f"Created {output}; visual comparison sheets: {work/'preview'}")
            return report
    except Timeout:
        raise RuntimeError("Another SlideTwin process is using this work directory") from None
    except Exception as exc:
        if isinstance(exc,(ProtocolError,ProviderError,LayoutError)) and 'document' in locals():
            from .best_effort import publish_best_effort
            with FileLock(work/'run.lock',timeout=0):
                log(f'Some steps remain incomplete ({exc}); exporting retained translations with local recovery')
                return publish_best_effort(source,output,work,document,selected,config,exc,
                                           values=locals().get('translations'),preview=preview,log=log,
                                           preflight=(placements,failures) if 'placements' in locals() else None)
        current = read_cache(work/'run.json')
        if current.get("status") == "running":
            write_json(work/"run.json", {"status": "failed", "reason": str(exc), "source_sha256": initial_hash,
                                       "selected_pages": selected, "final_output_published": False})
        raise
