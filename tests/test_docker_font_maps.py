"""Build font repairs preserve shaping except the proven unsafe digit lookup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unicodedata

import pytest

pytest.importorskip('fontTools')
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable, table__c_m_a_p
from fontTools.ttLib import newTable


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('docker_fonts', ROOT/'docker-fonts.py')
fonts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fonts)


def test_only_locl_unmapped_ascii_digit_substitutions_are_removed():
    font = TTFont()
    cmap = table__c_m_a_p()
    table = CmapSubtable.newSubtable(4)
    table.platformID, table.platEncID, table.language = 3, 1, 0
    table.cmap = {49: 'one', 50: 'two', 0xFF12: 'two.full', 0x4E2D: 'han', 0x2026: 'ellipsis'}
    cmap.tables = [table]
    font['cmap'] = cmap
    locl = {'one': 'one.locl', 'two': 'two.full', 'han': 'han.locl', 'ellipsis': 'ellipsis.locl'}
    tnum = {'one': 'one.tabular'}
    gsub = newTable('GSUB')
    gsub.table = NS(FeatureList=NS(FeatureRecord=[
        NS(FeatureTag='locl', Feature=NS(LookupListIndex=[0])),
        NS(FeatureTag='tnum', Feature=NS(LookupListIndex=[1]))]),
        LookupList=NS(Lookup=[NS(LookupType=1, SubTable=[NS(mapping=locl)]),
                             NS(LookupType=1, SubTable=[NS(mapping=tnum)])]))
    font['GSUB'] = gsub
    removed = fonts.repair_font_maps(font)
    assert [entry['unicode'] for entry in removed] == [49]
    assert locl == {'two': 'two.full', 'han': 'han.locl', 'ellipsis': 'ellipsis.locl'}
    assert tnum == {'one': 'one.tabular'}
    assert 'GSUB' in font


def test_full_pinned_font_digits_ascii_cjk_bold_subset_roundtrip(tmp_path):
    sources = [ROOT/'.local/fonts'/f'NotoSansSC-{style}.ttf' for style in ('Regular', 'Bold')]
    if not all(path.is_file() for path in sources):
        pytest.skip('Full local font cache absent; Docker build runs mandatory round-trip checks')
    output = []
    for path in sources:
        font = TTFont(path)
        glyf = font['glyf']
        original_order = list(font.getGlyphOrder())
        original = {name: glyf[name].compile(glyf) for name in original_order}
        original_metrics = {name: font['hmtx'].metrics[name] for name in original_order}
        original_gsub = font['GSUB'].table.LookupList.Lookup[10].SubTable[0].mapping.copy()
        removed = fonts.repair_font_maps(font)
        assert len(removed) == 10 and {entry['unicode'] for entry in removed} == set(range(48,58))
        assert all(glyf[name].compile(glyf) == data for name, data in original.items())
        assert all(font['hmtx'].metrics[name] == metrics for name, metrics in original_metrics.items())
        assert font['GSUB'].table.LookupList.Lookup[10].SubTable[0].mapping == {'ellipsis': original_gsub['ellipsis']}
        cmap = font.getBestCmap()
        assert len({cmap[code] for code in (0x2D, 0xAD, 0x2011)}) == 3
        for code in (0xAD, 0x2011):
            assert glyf[cmap[code]].compile(glyf) == glyf[cmap[0x2D]].compile(glyf)
            assert font['hmtx'].metrics[cmap[code]] == font['hmtx'].metrics[cmap[0x2D]]
        safe = tmp_path/path.name
        font.save(safe)
        output.append(safe)
    fonts.validate_font_roundtrip(output)
