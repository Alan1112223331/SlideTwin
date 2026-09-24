"""Unmodified Docling exports, reusable by the translation extraction stage."""
from pathlib import Path

import pymupdf as fitz

from .models import digest, read_cache, write_json


def convert_docling(source: Path, work: Path, selected: list[int], log=print) -> dict:
    from docling_core.types.doc import DoclingDocument

    work.mkdir(parents=True, exist_ok=True)
    sha = digest(source.read_bytes())
    raw = work / 'docling-document.json'
    cache = read_cache(work / 'extraction-key.json')
    reusable = (raw.is_file() and cache.get('source_sha256') == sha
                and cache.get('selected_pages') == selected and cache.get('docling_complete')
                and cache.get('docling_sha256') == digest(raw.read_bytes()))
    warnings = []
    if reusable:
        document = DoclingDocument.load_from_json(raw)
        log('Docling: reusing verified original extraction')
    else:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        subset = work / 'selected-source.pdf'
        with fitz.open(source) as original, fitz.open() as pdf:
            for number in selected:
                pdf.insert_pdf(original, from_page=number - 1, to_page=number - 1)
            pdf.save(subset, garbage=4, deflate=True)
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.table_structure_options.do_cell_matching = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        log(f'Docling: extracting {len(selected)} pages')
        result = converter.convert(subset, raises_on_error=False)
        status = str(result.status).split('.')[-1].lower()
        if status not in {'success', 'partial_success'}:
            raise RuntimeError('Docling conversion failed')
        document = result.document
        temp = work / 'docling-document.tmp.json'
        document.save_as_json(temp)
        temp.replace(raw)
        write_json(work / 'ocr-provenance.json', {
            'source_sha256': sha, 'selected_pages': selected,
            'pages': {str(selected[p.page_no - 1]): any(getattr(c, 'from_ocr', False) for c in p.cells)
                      for p in result.pages if 1 <= p.page_no <= len(selected)},
        })
        # Do not mark the enriched SlideTwin extraction as complete here.
        write_json(work / 'extraction-key.json', {
            'source_sha256': sha, 'selected_pages': selected, 'complete': False,
            'docling_complete': status == 'success', 'docling_sha256': digest(raw.read_bytes()),
        })
        if status != 'success':
            warnings.append('docling_partial_result')
    markdown = work / 'docling-document.md'
    try:
        temp = work / 'docling-document.tmp.md'
        document.save_as_markdown(temp)
        temp.replace(markdown)
    except Exception:
        markdown.unlink(missing_ok=True)
        warnings.append('docling_markdown_export_failed')
    return {'warnings': warnings, 'selected_pages': selected}
