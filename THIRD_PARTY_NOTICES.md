# Third-party notices

SlideTwin's original code is licensed under AGPL-3.0-only. Dependencies, fonts,
model weights and system packages retain their own licenses and copyright notices.
The SlideTwin license does not replace or relicense those components.

## PyMuPDF / MuPDF

SlideTwin uses PyMuPDF 1.28.2 and its MuPDF engine under their GNU Affero General
Public License v3 terms. Artifex also offers a separate commercial license;
SlideTwin does not include such a commercial grant.

- Project: https://github.com/pymupdf/PyMuPDF
- Licensing: https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright
- Upstream copyright and license texts are retained in the installed distribution.

## Docling

Docling 2.123.1 is distributed under the MIT license. Its upstream notices remain
in the installed distribution. Downloaded layout/OCR model weights may carry
separate licenses; consult their respective model repositories.

- Project and license: https://github.com/docling-project/docling

## Noto Sans SC fonts

The Docker build downloads Noto Sans SC from Google Fonts at commit
`a85815a42757630ce188fdad368c2dfc444d4773`, verifies SHA-256 digests, instantiates
regular and bold static fonts, and removes duplicate compatibility-character
aliases from Unicode mappings. These modified font files remain under the
SIL Open Font License 1.1, with upstream copyright notices retained.

- Source and OFL text: https://github.com/google/fonts/tree/a85815a42757630ce188fdad368c2dfc444d4773/ofl/notosanssc
- Reproduction script: `docker-fonts.py`
- Installed font license: `/usr/share/fonts/truetype/slidetwin/OFL.txt`

## Other dependencies

The Python dependencies are listed in `pyproject.toml`; tested Linux versions are
recorded in `requirements-docker.txt`. Each installed distribution retains its
own license files and metadata. Debian packages installed by the Dockerfile
include their notices under `/usr/share/doc/<package>/copyright`.

This document highlights components directly relevant to SlideTwin's packaging;
it is not a replacement for each dependency's complete license and notices.
