from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from html import escape
from pathlib import Path
import statistics
import re

import numpy as np
import pymupdf as fitz

from .config import Layout
from .extract import restore, immutable_math_label, native_lines, native_bullet, BULLETS
from .models import Document, Region, write_json
from .protocol import aligned_phrase
from .raster import raster_container


class LayoutError(RuntimeError):
    def __init__(self, message, region_id=None):
        super().__init__(message)
        self.region_id=region_id


@dataclass
class Placement:
    id: str
    page: int
    text: str
    frame: list[float]
    source_bbox: list[float]
    size: float
    scale: float
    align: str
    bold: bool
    color: int
    native: bool
    erase: list[list[float]]
    background: list[float] | None = None
    html_body: str = ""
    assets: dict[str, str] | None = None
    extractable_text: str = ""
    glyph_boxes: list[list[float]] | None = None
    shadow_boxes: list[list[float]] | None = None
    raster_patch: str | None = None
    raster_patch_box: list[float] | None = None
    background_box: list[float] | None = None
    rotate: int = 0


def intersects(a: fitz.Rect, b: fitz.Rect, epsilon=0.1) -> bool:
    c = a & b
    return not c.is_empty and c.width > epsilon and c.height > epsilon


def unicode_font() -> Path | None:
    for name in ['C:/Windows/Fonts/seguisym.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Apple Symbols.ttf']:
        if Path(name).is_file():return Path(name)
    return None


def extended_font() -> Path | None:
    for name in ['C:/Windows/Fonts/calibri.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(name).is_file():return Path(name)
    return None


def content_html(text: str, size: float, color: int, bold: bool, align: str, line_height: float, html_body: str = "") -> str:
    # Story measures a lowered inline image's box without the adjacent font's
    # descender. Reserve that space in layout, so fitted text really stays inside
    # its frame instead of colliding with the next source line.
    bottom = "0.22em" if '<img ' in html_body else "0"
    return (f'<div style="font-family:twin,latin,unicode,extended,symbols;font-size:{size:.4f}pt;color:#{color:06x};'
            f'font-weight:{700 if bold else 400};text-align:{align};line-height:{line_height};'
            f'white-space:normal;margin:0;padding:0 0 {bottom} 0;text-indent:0;">{html_body or escape(text)}</div>')


def has_orphan_cjk_line(page: fitz.Page) -> bool:
    lines=[''.join(s['text'] for s in line['spans']).strip() for block in page.get_text('dict')['blocks'] for line in block.get('lines',[])]
    return len(lines)>1 and bool(re.fullmatch(r'[\u3400-\u9fff]',lines[-1]))


def css_fonts(fonts: tuple[Path, Path]) -> tuple[str, fitz.Archive]:
    regular, bold = fonts
    archive = fitz.Archive()
    # Named byte entries handle non-ASCII paths and distinct font directories.
    archive.add((regular.read_bytes(), "regular.ttf"))
    archive.add((bold.read_bytes(), "bold.ttf"))
    archive.add((fitz.Font("helv").buffer, "latin.cff"))
    archive.add((fitz.Font("symb").buffer, "symbols.cff"))
    css = ("*{box-sizing:border-box;}body{margin:0;padding:0;}"
           "@font-face{font-family:twin;src:url(regular.ttf);font-weight:400;}"
           "@font-face{font-family:twin;src:url(bold.ttf);font-weight:700;}"
           "@font-face{font-family:latin;src:url(latin.cff);}"
           "@font-face{font-family:symbols;src:url(symbols.cff);}")
    fallback=unicode_font()
    if fallback:
        archive.add((fallback.read_bytes(),'unicode.ttf'))
        css += '@font-face{font-family:unicode;src:url(unicode.ttf);}'
    fallback=extended_font()
    if fallback:
        archive.add((fallback.read_bytes(),'extended.ttf'))
        css += '@font-face{font-family:extended;src:url(extended.ttf);}'
    return css, archive


def candidate_frame(region: Region, regions: list[Region], page: fitz.Page, grow=False) -> tuple[fitz.Rect, str]:
    box = fitz.Rect(region.bbox)
    # Find a true enclosing background panel (not an invented white mask).
    containers = [fitz.Rect(d["rect"]) for d in page.get_drawings() if d.get("fill") is not None
                  and (box in fitz.Rect(d["rect"]) + (-0.2, -0.2, 0.2, 0.2))
                  and fitz.Rect(d["rect"]).width > box.width+2 and fitz.Rect(d["rect"]).height > box.height+2]
    container = min(containers, key=lambda r: r.get_area()) if containers else page.rect
    bitmap_container=(raster_container(page,box) or raster_container(page,box,bridge_glyphs=True)) if not region.native else None
    if bitmap_container is not None:container=bitmap_container
    right_limit = min(container.x1-2, page.rect.x1-3)
    bottom_limit = min(container.y1-1, page.rect.y1-2)
    tight_bottom = None
    for other in regions:
        if other.id == region.id:
            continue
        b = fitz.Rect(other.bbox)
        if b.y0 < box.y1+0.1 and b.y1 > box.y0-0.1 and b.x0 >= box.x1-0.2:
            right_limit = min(right_limit, b.x0-1)
        if b.x0 < box.x1 and b.x1 > box.x0 and b.y0 >= box.y1-0.2:
            bottom_limit = min(bottom_limit, b.y0-1)
        if (b.x0 < box.x1 and b.x1 > box.x0 and
                box.y0+region.size*.55 < b.y0 < box.y1-.2):
            # Source font ascender boxes can overlap on tightly stacked labels.
            # Chinese glyphs occupy the full em: fit into the measured row pitch
            # instead of inheriting that overlap from the English font metrics.
            tight_bottom=min(tight_bottom if tight_bottom is not None else b.y0,b.y0-.5)
    # Formulas and numeric labels are not translation targets, but still constrain
    # expansion. Include them as obstacles instead of considering only regions.
    for word in page.get_text("words"):
        b = fitz.Rect(word[:4])
        if (b.tl+b.br)/2 in box + (-0.2, -0.2, 0.2, 0.2):
            continue
        if b.y0 < box.y1 and b.y1 > box.y0 and b.x0 >= box.x1-0.2:
            right_limit = min(right_limit, b.x0-1)
        if b.x0 < box.x1 and b.x1 > box.x0 and b.y0 >= box.y1-0.2:
            bottom_limit = min(bottom_limit, b.y0-1)
    # Nearby artwork bounds also constrain whitespace expansion. A background
    # enclosing the current text is its container, not an adjacent obstacle.
    graphics=[fitz.Rect(d['rect']) for d in page.get_drawings()]+[fitz.Rect(x['bbox']) for x in page.get_image_info()]
    for b in graphics:
        if box in b+(-1,-1,1,1):continue
        if b.y0<box.y1 and b.y1>box.y0 and b.x0>=box.x1+.2:
            right_limit=min(right_limit,b.x0-1)
    if grow and region.native and region.role=='page_footer' and box.x0>page.rect.width*.65:
        # A right-edge footer can use verified whitespace to its LEFT. This
        # changes geometry only; even repeated model wording remains intact.
        left=max(container.x0+3,box.x0-max(box.width*3,region.size*8))
        obstacles=[fitz.Rect(r.bbox) for r in regions if r.id!=region.id]
        obstacles += [fitz.Rect(w[:4]) for w in page.get_text('words') if not ((fitz.Rect(w[:4]).tl+fitz.Rect(w[:4]).br)/2 in box+(-.2,-.2,.2,.2))]
        obstacles += [b for b in graphics if not box in b+(-1,-1,1,1)]
        for b in obstacles:
            if b.y0<box.y1+region.size*.32 and b.y1>box.y0-region.size*.05 and b.x1<=box.x0+.2:
                left=max(left,b.x1+1)
        return fitz.Rect(left,box.y0-region.size*.05,box.x1,min(bottom_limit,box.y0+max(box.height+region.size*.32,region.size*1.28))), 'right'
    # Keep the source left edge exactly: it includes nesting/hanging indentation.
    # Allow bounded growth into confirmed whitespace, not across adjacent objects.
    width = right_limit-box.x0 if grow else box.width
    if grow and not region.native and bitmap_container is None:
        width=min(width,box.width*1.35)
    height = min(bottom_limit-box.y0, max(box.height + region.size*0.32, region.size*1.28))
    frame = fitz.Rect(box.x0, box.y0-region.size*0.05, box.x0+max(width, box.width), box.y0+max(height, box.height))
    if tight_bottom is not None:frame.y1=min(frame.y1,tight_bottom)
    # Native centered headings/labels retain their original center. Source list
    # text never receives a synthetic center or extra indent.
    align = "left"
    near_center = abs((box.x0+box.x1)-(container.x0+container.x1)) < max(3, container.width*0.012)
    rows = []
    for glyph in sorted(region.erase, key=lambda b:(b[1],b[0])):
        b = fitz.Rect(glyph)
        row = next((r for r in rows if abs(r.y0-b.y0) < region.size*0.45), None)
        if row is None:
            rows.append(b)
        else:
            row.include_rect(b)
    left_aligned_rows = len(rows)>1 and max(r.x0 for r in rows)-min(r.x0 for r in rows)<max(2,region.size*0.35)
    centered_rows = len(rows)>1 and max((r.x0+r.x1)/2 for r in rows)-min((r.x0+r.x1)/2 for r in rows)<max(2,region.size*0.35)
    centered = near_center and not left_aligned_rows and (bool(containers) or centered_rows or region.role in {"title","section_header"})
    if centered and region.role != "list_item":
        align = "center"
        half = min((frame.width)/2, (box.x0+box.x1)/2-container.x0-2, container.x1-2-(box.x0+box.x1)/2)
        frame.x0 = (box.x0+box.x1)/2-half
        frame.x1 = (box.x0+box.x1)/2+half
    if bitmap_container is not None and abs((box.x0+box.x1)-(container.x0+container.x1))<container.width*.25:
        # Centered chart labels can use the cell's width, never its neighbor.
        frame.x0=container.x0+1;frame.x1=container.x1-1;align='center'
    return frame & page.rect, align


def raster_background(page: fitz.Page, box: fitz.Rect) -> list[float]:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=box + (-1, -1, 1, 1), alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(float)
    border = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3), a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # A varied border indicates touching rules, chart strokes or texture. Do not
    # erase it and pretend preservation succeeded.
    if np.percentile(np.max(np.abs(border-bg), axis=1), 90) > 18:
        raise LayoutError("OCR label touches a non-uniform background or graphic; manual/local repair required")
    if np.mean(np.max(np.abs(a-bg), axis=2) > 35) > 0.42:
        raise LayoutError("OCR region is too dense for safe local background restoration")
    return [float(x/255) for x in bg]


def symbolic_quantity(text: str) -> bool:
    """A diagram's dimensions and timing inequalities stay original artwork."""
    return immutable_math_label(text)


def visible_native_ink(page: fitz.Page,region: Region) -> bool:
    """Do not expose template text hidden behind an opaque source shape."""
    scale=4
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=fitz.Rect(region.bbox),alpha=False)
    import cv2
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3].astype(float)
    colors=[region.color]+[s['color'] for s in region.inline_styles]
    count=0
    for color in set(colors):
        rgb=np.array([(color>>16)&255,(color>>8)&255,color&255])
        mask=(np.max(np.abs(a-rgb),axis=2)<30).astype(np.uint8)
        n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
        count+=sum(area for x,y,w,h,area in stats[1:] if w<region.size*scale*4 and
                   region.size*scale*.25<h<region.size*scale*2.2 and area/(w*h)>.12)
    return count>max(4,pix.width*pix.height*.002)


def fixed_native_suffix(region: Region,lines,regions=()):
    """Keep an identifier whose subscript/range is outside an OCR text box."""
    if region.native and region.protected_assets:
        trailing=re.search(r'([({]?⟦P\d+⟧)$',region.source)
        if trailing:
            token=re.search(r'⟦P\d+⟧',trailing[1])[0];asset=region.protected_assets.get(token)
            if asset:
                ab=fitz.Rect(asset['bbox'])
                split=any(other.id!=region.id and other.protected_assets and
                          (ab & fitz.Rect(other.bbox)).get_area()>0 for other in regions)
                owned=[fitz.Rect(b)+(-.03,-.03,.03,.03) for b in region.erase]
                split=split or any(c['size']<region.size*.9 and not c['c'].isspace()
                    and (fitz.Rect(c['bbox']).tl+fitz.Rect(c['bbox']).br)/2 in ab+(-.2,-.2,.2,region.size*.4)
                    and not any((fitz.Rect(c['bbox']).tl+fitz.Rect(c['bbox']).br)/2 in b for b in owned)
                    for line in lines for c in line['chars'])
                if split:
                    preserved=ab+(-.2,-.2,.2,.2)
                    if trailing[1][0] in '({':
                        brackets=[c for line in lines for c in line['chars'] if c['c']==trailing[1][0] and
                                  0<=ab.x0-c['bbox'][2]<region.size and abs(c['bbox'][1]-ab.y0)<region.size]
                        for c in brackets:preserved.include_rect(fitz.Rect(c['bbox']))
                    erase=[b for b in region.erase if (fitz.Rect(b).tl+fitz.Rect(b).br)/2 not in preserved]
                    if erase and len(erase)<len(region.erase):
                        box=fitz.Rect(erase[0])
                        for b in erase[1:]:box.include_rect(b)
                        return replace(region,bbox=list(box),erase=erase),trailing[1]
    if not region.native or region.inline_styles or region.protected_assets:return region,None
    literal=restore(region,region.source);match=re.search(r'\b([A-Za-z]|\d+)$',literal)
    if not match or len(match[1])!=1:return region,None
    box=fitz.Rect(region.bbox);chars=[c for line in lines for c in line['chars'] if not c['c'].isspace()]
    owned=[c for c in chars if (fitz.Rect(c['bbox']).tl+fitz.Rect(c['bbox']).br)/2 in box]
    if not owned:return region,None
    last=max(owned,key=lambda c:(round(c['origin'][1]/region.size),c['bbox'][0]))
    if last['c']!=match[1]:return region,None
    following=[c for c in chars if -.5<=c['bbox'][0]-last['bbox'][2]<region.size*.35 and
               abs(c['origin'][1]-last['origin'][1])<region.size*.4 and c not in owned]
    if not any((c['c'].isdigit() and c['size']<last['size']*.9) or c['c'] in '–−-' for c in following):return region,None
    token=next((k for k,v in region.protected.items() if region.source.endswith(k) and v==match[1]),match[1])
    erase=[b for b in region.erase if abs(b[0]-last['bbox'][0])>.1 or abs(b[1]-last['bbox'][1])>.1]
    if len(erase)==len(region.erase):return region,None
    box.x1=last['bbox'][0]-.4
    return replace(region,bbox=list(box),erase=erase),token


def raster_typography(page: fitz.Page,region: Region) -> tuple[Region,int,bool]:
    """Estimate the actual source glyph height/color, including rotated axes.

    Docling paragraph boxes contain several lines; their total height is not a
    font size. Connected source ink components supply the typographic scale.
    """
    import cv2
    box=fitz.Rect(region.bbox);scale=4
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=box+(-1,-1,1,1),alpha=False)
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
    border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]])
    bg=np.median(border,axis=0)
    ink=(np.max(np.abs(a.astype(float)-bg),axis=2)>55).astype(np.uint8)
    rotate=90 if box.height>box.width*2 and len(re.sub(r'[^A-Za-z]','',region.source))>=4 else 0
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    components=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area>=5 and w>=2 and h>=3 and w<max(12,pix.width*.98) and h<max(12,pix.height*.95):
            components.append((i,x,y,w,h,area))
    # Scanned letters can touch and form a whole-word component. Its width is
    # not font height; excluding wide words leaves only tiny antialias specks.
    sizes=[(w if rotate else h)/scale for _,x,y,w,h,area in components
           if min(w,h)>=3 and (w if rotate else h)<max(12,region.size*scale*1.6)
           and (h if rotate else w)<20*(w if rotate else h)]
    size=min(region.size,float(np.percentile(sizes,85))/.72) if sizes else region.size
    size=max(4,size)
    if '/prose/' in region.docling_ref and components:
        # Formula OCR line boxes include the neighboring equation's subscripts.
        # Align prose to its own measured ink, not that oversized line box.
        x0=min(c[1] for c in components);y0=min(c[2] for c in components)
        x1=max(c[1]+c[3] for c in components);y1=max(c[2]+c[4] for c in components)
        box=fitz.Rect((pix.x+x0)/scale,(pix.y+y0)/scale,(pix.x+x1)/scale,(pix.y+y1)/scale)
    # A solid square separated at the left of the first body line is a source
    # bullet, even if OCR put the square in the middle of its returned sentence.
    bullet=False
    if not rotate and components:
        first=min(components,key=lambda c:c[1]);i,x,y,w,h,area=first
        right=[c for c in components if c[1]>=x+w and c[2]<y+h and c[2]+c[4]>y]
        gap=min((c[1]-(x+w) for c in right),default=0)
        rest=right if gap>size*scale*.45 else []
        literal_marker=bool(re.search('[■□▪●]',region.source))
        # OCR often drops hollow list squares. A separate colored, hollow square
        # before the body is source artwork: retain it and the measured indent.
        hollow=False
        if rest and .8<w/max(1,h)<1.25:
            contours,hierarchy=cv2.findContours((labels==i).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            enclosed=hierarchy is not None and any(hh[3]>=0 for hh in hierarchy[0])
            first_color=np.median(a[labels==i],axis=0)
            body_color=np.median(a[labels==rest[0][0]],axis=0)
            hollow=enclosed and np.max(np.abs(first_color-body_color))>30
        solid=literal_marker and area/(w*h)>.6 and x<12
        if .6<w/max(1,h)<1.6 and area/(w*h)>.3 and rest and x<max(24,size*scale*.7) and (solid or hollow):
            next_x=min(c[1] for c in rest);box.x0=(pix.x+next_x)/scale;bullet=True
    pixels=a[ink.astype(bool)]
    color=region.color
    if len(pixels):
        # Anti-aliased edges outnumber solid ink in tiny scanned type. Measure
        # its foreground core, otherwise black labels become pale gray Chinese.
        luma=pixels.mean(axis=1)
        dark_text=np.median(luma)<np.mean(bg)
        threshold=np.percentile(luma,20 if dark_text else 80)
        core=pixels[luma<=threshold] if dark_text else pixels[luma>=threshold]
        if len(core)>=3:pixels=core
        # Quantized mode is robust to anti-aliasing and a minority accent color.
        quant=pixels//24;unique,counts=np.unique(quant,axis=0,return_counts=True)
        bucket=unique[counts.argmax()];rgb=np.median(pixels[np.all(quant==bucket,axis=1)],axis=0).astype(int)
        color=int(rgb[0])*65536+int(rgb[1])*256+int(rgb[2])
    return replace(region,bbox=list(box),size=size,color=color),rotate,bullet


def raster_label_patch(page: fitz.Page, region: Region, work: Path, neighbors: list[Region] | None = None) -> tuple[str,list[float],int]:
    """Remove connected glyph pixels, preserving nearby rules and texture.

    High-contrast neutral glyphs are isolated in a padded crop. Components
    touching the crop edge are likely graphic strokes and remain untouched.
    """
    import cv2
    scale=4
    box=fitz.Rect(region.bbox)
    padding=max(5,min(12,region.size*1.5))
    crop=(box+(-padding,-padding,padding,padding)) & page.rect
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=crop,alpha=False)
    pixels=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3].copy()
    gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    border=np.concatenate([pixels[:5].reshape(-1,3),pixels[-5:].reshape(-1,3),pixels[:,:5].reshape(-1,3),pixels[:,-5:].reshape(-1,3)])
    background=np.median(border,axis=0);bg_luma=float(np.mean(background))
    neutral=np.ptp(pixels.astype(np.int16),axis=2)<110
    dark=(gray<min(215,bg_luma-35)) & neutral
    light=(gray>max(135,bg_luma+50)) & neutral
    measured_color=raster_typography(page,region)[0].color if region.color==0 else region.color
    ink_color=np.array([(measured_color>>16)&255,(measured_color>>8)&255,measured_color&255])
    # Typography has already measured the glyph color. A dark photo can contain
    # more dark texture than white letters; pixel counts alone reverse polarity.
    light_text=float(ink_color.mean())>bg_luma+35
    raw=light if light_text else dark
    if np.ptp(ink_color)>50:
        raw=(np.max(np.abs(pixels.astype(int)-ink_color),axis=2)<80) & (np.max(np.abs(pixels.astype(float)-background),axis=2)>35)
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype(np.uint8),8)
    mask=np.zeros(gray.shape,np.uint8);kept=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        center=fitz.Point((pix.x+x+w/2)/scale,(pix.y+y+h/2)/scale)
        page_edge_glyph=(abs(crop.y1-page.rect.y1)<.01 and center in box
                         and w<region.size*scale*1.2 and h<region.size*scale*1.5)
        if x<2 or y<2 or x+w>=pix.width-2 or (y+h>=pix.height-2 and not page_edge_glyph):continue
        if center not in box+(-region.size*.8,-region.size*.8,region.size*.8,region.size*.8):continue
        if center not in box and any(other.id!=region.id and center in fitz.Rect(other.bbox) for other in (neighbors or [])):continue
        if area<3 or h>box.height*scale*1.15 or w>max(h*7,box.width*scale*.9):continue
        if w>h*8 or h>w*12:continue
        mask[labels==i]=255;kept.append(i)
    coverage=np.mean(mask>0)
    if not kept or not .003<coverage<.4:
        raise LayoutError('Cannot isolate raster label glyphs safely')
    color=np.array([255,255,255] if light_text else [(region.color>>16)&255,(region.color>>8)&255,region.color&255])
    protected_strokes=raw & (mask==0)
    # Low-contrast edges can be tiny disconnected components of the SAME letter.
    # Do not protect those fragments inside the label, or inpainting leaves halos.
    ix0=max(0,round(box.x0*scale)-pix.x);iy0=max(0,round(box.y0*scale)-pix.y)
    ix1=min(pix.width,round(box.x1*scale)-pix.x);iy1=min(pix.height,round(box.y1*scale)-pix.y)
    protected_strokes[iy0:iy1,ix0:ix1]=False
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if w>h*8 or h>w*12:protected_strokes[labels==i]=True
    mask=cv2.dilate(mask,np.ones((7,7),np.uint8),iterations=2)
    mask[protected_strokes]=0
    cleaned=cv2.inpaint(pixels,mask,7,cv2.INPAINT_TELEA)
    # Flat diagram/table fills can be restored exactly. Inpainting a white
    # letter beside a white rule otherwise propagates that rule into a cloud.
    ring=cv2.dilate(mask,np.ones((13,13),np.uint8),iterations=1)>0
    samples=pixels[ring & (mask==0) & ~raw]
    if len(samples):
        quant=samples//12;colors,counts=np.unique(quant,axis=0,return_counts=True)
        winner=colors[counts.argmax()];same=np.all(quant==winner,axis=1)
        if same.mean()>.7:
            fill=np.median(samples[same],axis=0).astype(np.uint8)
            cleaned[mask>0]=fill
    folder=work/'raster-assets';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{region.id}.png'
    from PIL import Image
    # Transparent outside changed glyph pixels: overlapping label patches cannot
    # reintroduce an original word from their own source crop.
    Image.fromarray(np.dstack([cleaned,mask])).save(path)
    actual=[pix.x/scale,pix.y/scale,(pix.x+pix.width)/scale,(pix.y+pix.height)/scale]
    return str(path.resolve()),actual,int(color[0])*65536+int(color[1])*256+int(color[2])


def text_shadow_images(page: fitz.Page, entries: list[Placement]) -> list[tuple[int, list[float], Placement]]:
    """Recognize translucent monochrome text shadows, never opaque artwork.

    Every nontrivial mask pixel must lie near native glyphs being translated.
    Removing the image object exposes its original background, avoiding paint.
    """
    result = []
    native = [p for p in entries if p.native]
    for info in page.get_images(full=True):
        xref, smask = info[:2]
        if not smask:
            continue
        pix = fitz.Pixmap(page.parent, xref)
        colors = np.frombuffer(pix.samples, np.uint8).reshape(-1, pix.n)[:, :3]
        if np.max(colors) > 25 or np.any(np.ptp(colors.astype(np.int16), axis=0) > 2):
            continue
        mask = fitz.Pixmap(page.parent, smask)
        alpha = np.frombuffer(mask.samples, np.uint8).reshape(mask.height, mask.width, mask.n)[:, :, 0]
        if not 5 < int(alpha.max()) < 180 or len(np.unique(alpha)) < 8 or not .01 < float(np.mean(alpha>3)) < .45:
            continue
        for box in page.get_image_rects(xref):
            relevant = [p for p in native if intersects(box, fitz.Rect(p.source_bbox))]
            if not relevant:
                continue
            allowed = np.zeros_like(alpha, dtype=bool)
            for entry in relevant:
                for glyph in entry.erase:
                    r = (fitz.Rect(glyph) + (-entry.size*.8, -entry.size*.8, entry.size*.8, entry.size*.8)) & box
                    if r.is_empty:
                        continue
                    x0=max(0,int((r.x0-box.x0)/box.width*mask.width)); x1=min(mask.width,int((r.x1-box.x0)/box.width*mask.width)+1)
                    y0=max(0,int((r.y0-box.y0)/box.height*mask.height)); y1=min(mask.height,int((r.y1-box.y0)/box.height*mask.height)+1)
                    allowed[y0:y1,x0:x1]=True
            ink=alpha>3
            if ink.any() and np.count_nonzero(ink & ~allowed)/np.count_nonzero(ink) < .005:
                owner=max(relevant,key=lambda p:(box & fitz.Rect(p.source_bbox)).get_area())
                result.append((xref,list(box),owner))
    return result


def styled_html(region: Region, value: str, translations: dict, replacements: dict[str,str], warnings=None) -> str:
    """Apply phrase ranges to prose AND atomic original mathematical glyphs."""
    plain=restore(region,value);ranges=[]
    for i,style in enumerate(region.inline_styles):
        phrase=translations.get(f'{region.id}_s{i}','')
        match=aligned_phrase(plain,phrase) if phrase else None
        if match is None:
            if warnings is not None:warnings.append({'page':region.page,'id':f'{region.id}_s{i}','kind':'unmatched_emphasis','action':'Use base text style; retain model sentence unchanged'})
            continue
        start=plain.find(match);ranges.append((start,start+len(match),style))
    def wrap(body,start,end):
        for lo,hi,style in ranges:
            if start<hi and end>lo:
                body=(f'<span style="color:#{style["color"]:06x};font-weight:{700 if style["bold"] else 400};'
                      f'font-style:{"italic" if style["italic"] else "normal"};font-size:{style.get("size_ratio",1):.4f}em;">{body}</span>')
                if style.get('break_before') and start==lo:body='<br>'+body
                if style.get('break_after') and end==hi:body+='<br>'
        return body
    out=[];position=0
    for part in re.split(r'(⟦P\d+⟧)',value):
        if part in region.protected:
            actual=region.protected[part];end=position+len(actual)
            out.append(wrap(replacements.get(part,escape(actual)),position,end));position=end
        else:
            boundaries=sorted({position,position+len(part)}|{x for lo,hi,_ in ranges for x in [lo,hi] if position<x<position+len(part)})
            for start,end in zip(boundaries,boundaries[1:]):out.append(wrap(escape(part[start-position:end-position]),start,end))
            position+=len(part)
    return ''.join(out)


def build_plan(source: Path, document: Document, translations: dict[str, str], selected: list[int], layout: Layout, work: Path) -> tuple[list[Placement], list[dict]]:
    fitz.TOOLS.set_small_glyph_heights(True)
    fonts = layout.fonts()
    css, archive = css_fonts(fonts)
    font_objects = [fitz.Font(fontfile=str(p)) for p in fonts]
    fallback_fonts = [fitz.Font("helv"), fitz.Font("symb")]
    if unicode_font():fallback_fonts.append(fitz.Font(fontfile=str(unicode_font())))
    if extended_font():fallback_fonts.append(fitz.Font(fontfile=str(extended_font())))
    supported_rotations={r.id for p in document.pages for r in p.regions if any(abs(r.direction[0]-x)<.02 and abs(r.direction[1]-y)<.02 for x,y in [(0,-1),(-1,0),(0,1)])}
    placements, failures = [], [d for d in document.diagnostics if d.get("blocking") and d.get("page") in selected
                               and not (d.get('kind')=='rotated_text' and d.get('id') in supported_rotations)]
    warnings=[]
    with fitz.open(source) as src:
        for number in selected:
            page = src[number-1]
            regions = document.pages[number-1].regions
            lines=native_lines(page)
            for region in regions:
                try:
                    if region.native and not visible_native_ink(page,region):continue
                    region,fixed_suffix=fixed_native_suffix(region,lines,regions)
                    rotate=0;raster_bullet=False
                    if not region.native:
                        region,rotate,raster_bullet=raster_typography(page,region)
                    first=region.source[:1]
                    source_bullet=first if first and (first in '✓✔☑❑' or '\ue000'<=first<='\uf8ff') else ''
                    if region.native and region.source.startswith('- '):source_bullet='-'
                    if region.native and not source_bullet:
                        b=fitz.Rect(region.bbox)
                        adjacent=[c for line in lines for c in line['chars'] if native_bullet(c) and
                                  0<=b.x0-c['bbox'][2]<region.size*1.8 and abs(c['bbox'][1]-b.y0)<region.size*.5]
                        if adjacent:source_bullet=adjacent[0]['c']
                    if region.native and source_bullet:
                        marks=[fitz.Rect(c['bbox']) for line in native_lines(page) for c in line['chars']
                               if c['c']==source_bullet and fitz.Point(c['origin']) in fitz.Rect(region.bbox)+(-1,-1,1,1)
                               and (source_bullet!='-' or c['bbox'][0]<region.bbox[0]+region.size*.6)]
                        body=[b for b in region.erase if not any((fitz.Rect(b).tl+fitz.Rect(b).br)/2 in mark for mark in marks)]
                        if marks and body:
                            body_box=fitz.Rect(body[0])
                            for b in body[1:]:body_box.include_rect(b)
                            region=replace(region,bbox=list(body_box),erase=body)
                    if symbolic_quantity(restore(region,region.source)):
                        continue
                    if region.id not in translations:
                        failures.append({"page": number, "id": region.id, "kind": "missing_translation"})
                        continue
                    text = restore(region, translations[region.id])
                    fixed_literal=restore(region,fixed_suffix) if fixed_suffix else None
                    if fixed_literal and text.rstrip().endswith(fixed_literal):text=text.rstrip()[:-len(fixed_literal)].rstrip()
                    compact_label=region.source.rstrip().endswith(':') and len(text)<8 and region.native
                    if compact_label:text=text.replace('：',':')
                    if source_bullet:text=text.lstrip(source_bullet+BULLETS+'□■❑ ')
                    if raster_bullet:text=re.sub('[■□▪●]','',text).strip()
                    if text.strip() == restore(region, region.source).strip():
                        # Accepted unchanged identifiers preserve the actual PDF object.
                        continue
                    if abs(region.direction[0]-1) > 0.02 or abs(region.direction[1]) > 0.02:
                        rotation=next((angle for x,y,angle in [(0,-1,90),(-1,0,180),(0,1,270)] if abs(region.direction[0]-x)<.02 and abs(region.direction[1]-y)<.02),None)
                        if rotation is None:
                            failures.append({'page':number,'id':region.id,'kind':'unsupported_text_rotation'})
                            continue
                        rotate=rotation
                    extractable = translations[region.id]
                    if fixed_suffix and extractable.rstrip().endswith(fixed_suffix):extractable=extractable.rstrip()[:-len(fixed_suffix)].rstrip()
                    if compact_label:extractable=extractable.replace('：',':')
                    if source_bullet:extractable=extractable.lstrip(source_bullet+BULLETS+'□■❑ ')
                    if raster_bullet:extractable=re.sub('[■□▪●]','',extractable).strip()
                    if region.native and region.inline_styles:
                        styles=[]
                        rb=fitz.Rect(region.bbox)
                        local_lines=[line for line in lines if any(fitz.Point(c['origin']) in rb+(-1,-1,1,1) for c in line['chars'])]
                        texts=[''.join(c['c'] for c in line['chars']).strip() for line in local_lines]
                        norm=lambda x:re.sub(r'\s+','',x)
                        for style in region.inline_styles:
                            match=next(((i,j) for i in range(len(texts)) for j in range(i+1,min(len(texts),i+6)+1)
                                        if norm(' '.join(texts[i:j]))==norm(style['source'])),None)
                            extra={}
                            if match:
                                i,j=match;sizes=[c['size'] for line in local_lines[i:j] for c in line['chars'] if not c['c'].isspace()]
                                baseline=lambda row:statistics.median(c['origin'][1] for c in row['chars'])
                                # Overprinted shadows are two PDF lines at one
                                # baseline, not two physical text rows.
                                extra={'break_before':i>0 and baseline(local_lines[i])-baseline(local_lines[i-1])>region.size*.55,
                                       'break_after':j<len(texts) and baseline(local_lines[j])-baseline(local_lines[j-1])>region.size*.55,
                                       'size_ratio':statistics.median(sizes)/region.size if sizes else 1}
                            styles.append({**style,**extra})
                        region=replace(region,inline_styles=styles)
                    html_body = escape(extractable)
                    token_text=extractable
                    replacements={}
                    assets = {}
                    for token, value in region.protected.items():
                        if token in region.protected_assets:
                            asset = region.protected_assets[token]
                            asset_dir = work/"math-assets"
                            asset_dir.mkdir(parents=True, exist_ok=True)
                            name = region.id + "-" + token[1:-1] + ".png"
                            path = asset_dir/name
                            box = fitz.Rect(asset["bbox"])
                            # An inline function and its native subscript form one
                            # atomic image; MuPDF can otherwise break at image edges
                            # even inside a CSS nowrap span.
                            function=re.search(r'\b(log|exp|sin|cos|tan|softmax|sigmoid)\(\s*'+re.escape(token)+r'\s*\)',token_text)
                            if function:
                                baseline=asset['bbox'][3]-asset['baseline_down']
                                same=[c for line in lines for c in line['chars'] if abs(c['origin'][1]-baseline)<region.size*.35]
                                left=sorted([c for c in same if box.x0-region.size*6<c['bbox'][0]<box.x0],key=lambda c:c['bbox'][0])
                                left_text=''.join(c['c'] for c in left)
                                prefix=re.search(re.escape(function[1])+r'\(\s*$',left_text)
                                right=sorted([c for c in same if -.2<=c['bbox'][0]-box.x1<region.size],key=lambda c:c['bbox'][0])
                                close=next((c for c in right if not c['c'].isspace()),None)
                                if prefix and close and close['c']==')':
                                    closings=[close];end=function.end()
                                    trailing=[c for c in right if not c['c'].isspace()]
                                    if (end<len(token_text) and token_text[end] in ')）' and len(trailing)>1
                                            and trailing[1]['c']==')'):
                                        closings.append(trailing[1]);end+=1
                                    original_function=token_text[function.start():end]
                                    for c in left[prefix.start():]+closings:box.include_rect(fitz.Rect(c['bbox']))
                                    token_text=token_text[:function.start()]+token+token_text[end:]
                                    extractable=extractable.replace(original_function,token,1)
                            page.get_pixmap(clip=box, dpi=360, alpha=True).save(path)
                            archive.add((path.read_bytes(), name))
                            assets[name] = str(path.resolve())
                            replacements[token]=f'<img src="{name}" style="width:{box.width}pt;height:{box.height}pt;vertical-align:-{asset["baseline_down"]}pt;">'
                            extractable = extractable.replace(token, "")
                        else:
                            replacements[token]=escape(value)
                            extractable = extractable.replace(token, value)
                    html_body=styled_html(region,token_text,translations,replacements,warnings)
                    # Function names, an inline original variable and closing
                    # parentheses form one mathematical unit for line breaking.
                    html_body=re.sub(r'\b(?:log|exp|sin|cos|tan|softmax|sigmoid)\((?:[^<>]|<img [^>]+>){1,400}?\)[）)]?',
                                     lambda m:'<span style="white-space:nowrap">'+m[0]+'</span>',html_body)
                    # Keep a short CJK label suffix together (e.g. INT8 + 50% 剪枝),
                    # avoiding a single orphan Chinese character on the next line.
                    suffix = re.search(r"[\u3400-\u9fff]{2,6}$", text)
                    if suffix and len(re.findall(r"[\u3400-\u9fff]", text)) <= 4 and re.search(r'(?:INT|FP)\d+|%',text) and fitz.Rect(region.bbox).height > region.size*1.5:
                        escaped_suffix = escape(suffix.group())
                        if html_body.endswith(escaped_suffix):
                            html_body = html_body[:-len(escaped_suffix)] + '<span style="white-space:nowrap">' + escaped_suffix + '</span>'
                    font = font_objects[int(region.bold)]
                    missing = sorted({ch for ch in extractable if not ch.isspace() and not any(f.has_glyph(ord(ch)) for f in [font, *fallback_fonts])})
                    if missing:
                        failures.append({"page": number, "id": region.id, "kind": "missing_font_glyph", "characters": missing})
                        continue
                    frame, align = candidate_frame(region, regions, page)
                    if rotate and region.native:frame=fitz.Rect(region.bbox)+(-.5,-.5,.5,.5)
                    if fixed_suffix and not region.protected_assets:align='right'
                    background = None
                    background_box = None
                    patch=patch_box=None
                    color=region.color
                    if region.native and region.role=='page_footer' and region.inline_styles:
                        # Some templates overprint a white duplicate and dark text.
                        # Use a visible source color as the base, never white ink on
                        # a white margin for an unmatched/repeated translated phrase.
                        pix=page.get_pixmap(clip=fitz.Rect(region.bbox)+(-2,-2,2,2),alpha=False)
                        pixels=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
                        bg=np.median(np.concatenate([pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]]),axis=0)
                        rgb=lambda c:np.array([(c>>16)&255,(c>>8)&255,c&255])
                        if max(abs(rgb(color)-bg))<30:
                            visible=[s['color'] for s in region.inline_styles if max(abs(rgb(s['color'])-bg))>50]
                            if visible:color=visible[0]
                    try:
                        if not region.native:
                            cell=raster_container(page,fitz.Rect(region.bbox)) or raster_container(page,fitz.Rect(region.bbox),bridge_glyphs=True)
                            precise_label=region.docling_ref.endswith('/caption-body') or region.docling_ref.startswith('raster/case/')
                            if precise_label:
                                patch,patch_box,color=raster_label_patch(page,region,work,regions)
                            elif cell is not None:
                                # The component proves the flat cell interior; stay
                                # inside it so table rules and connectors survive.
                                clean=(fitz.Rect(region.bbox)+(-1,-1,1,1)) & (cell+(.15,.15,-.15,-.15))
                                pix=page.get_pixmap(matrix=fitz.Matrix(3,3),clip=cell,alpha=False)
                                pixels=np.frombuffer(pix.samples,np.uint8).reshape(-1,pix.n)[:,:3]
                                colors,counts=np.unique(pixels//12,axis=0,return_counts=True)
                                same=np.all(pixels//12==colors[counts.argmax()],axis=1)
                                background=(np.median(pixels[same],axis=0)/255).tolist();background_box=list(clean)
                            else:
                                try:
                                    background=raster_background(page,fitz.Rect(region.bbox))
                                    box=fitz.Rect(region.bbox)
                                    if not rotate and len(region.source)<16 and box.height>box.width*.7:
                                        patch,patch_box,color=raster_label_patch(page,region,work,regions)
                                        background=None
                                except LayoutError:
                                    patch,patch_box,color=raster_label_patch(page,region,work,regions)
                        with fitz.open() as probe:
                            test = probe.new_page(width=page.rect.width, height=page.rect.height)
                            spare, scale = test.insert_htmlbox(frame, content_html(text, region.size, color, region.bold, align, layout.line_height, html_body),
                                                              css=css, archive=archive, scale_low=layout.min_font_scale,rotate=rotate)
                            glyph_boxes = [list(w[:4]) for w in test.get_text("words")]
                            orphan=has_orphan_cjk_line(test)
                        if spare < 0 or scale < 0.97 or orphan:
                            grown, grown_align = candidate_frame(region, regions, page, grow=True)
                            if rotate and region.native:grown,grown_align=frame,align
                            if fixed_suffix and not region.protected_assets:grown_align='right'
                            with fitz.open() as probe:
                                test = probe.new_page(width=page.rect.width, height=page.rect.height)
                                grown_spare, grown_scale = test.insert_htmlbox(grown, content_html(text, region.size, color, region.bold, grown_align, layout.line_height, html_body),
                                                                               css=css, archive=archive, scale_low=layout.min_font_scale,rotate=rotate)
                                if grown_spare >= 0 and (spare < 0 or grown_scale > scale or (orphan and not has_orphan_cjk_line(test))):
                                    frame, align, spare, scale = grown, grown_align, grown_spare, grown_scale
                                    glyph_boxes = [list(w[:4]) for w in test.get_text("words")]
                        if spare < 0:
                            raise LayoutError("Translation cannot fit above minimum readable font scale")
                        if any(fitz.Rect(b) not in frame + (-0.75, -0.75, 0.75, 0.75) for b in glyph_boxes):
                            raise LayoutError("Rendered glyph extends beyond its allocated text area")
                        placements.append(Placement(region.id, number, text, list(frame), region.bbox, region.size, scale, align,
                                                    region.bold, color, region.native, region.erase, background, html_body, assets, extractable, glyph_boxes,
                                                    raster_patch=patch,raster_patch_box=patch_box,background_box=background_box,rotate=rotate))
                    except (LayoutError, ValueError) as exc:
                        failures.append({"page": number, "id": region.id, "kind": "layout_blocked", "reason": str(exc)})
                except (LayoutError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    failures.append({'page':number,'id':region.id,'kind':'region_preflight_failed','reason':str(exc)})
        for number in selected:
            placed = [p for p in placements if p.page == number]
            for _, box, owner in text_shadow_images(src[number-1], placed):
                owner.shadow_boxes = (owner.shadow_boxes or []) + [box]
            for i, a in enumerate(placed):
                for b in placed[i+1:]:
                    if any(intersects(fitz.Rect(x), fitz.Rect(y), epsilon=0.6) for x in (a.glyph_boxes or []) for y in (b.glyph_boxes or [])):
                        failures.append({"page": number, "kind": "translated_text_overlap", "ids": [a.id, b.id]})
    write_json(work/"layout-plan.json", {"placements": [asdict(x) for x in placements], "failures": failures,"warnings":warnings})
    return placements, failures


def clear_native_bitmap_replicas(source_page,edited_page,entries):
    """Remove a raster copy printed directly underneath an editable label.

    Some figures contain both native text and the identical word baked into an
    overlapping image. Require source-color/shape agreement and a uniform
    background before covering the residual copy; never erase arbitrary art.
    """
    import cv2
    images=[fitz.Rect(i['bbox']) for i in source_page.get_image_info()]
    for entry in entries:
        box=fitz.Rect(entry.source_bbox)
        if not any((box & image).get_area()>box.get_area()*.8 for image in images):continue
        if any((fitz.Rect(w[:4]) & box).get_area()>1 for w in edited_page.get_text('words')):continue
        try:bg=np.array(raster_background(edited_page,box))*255
        except LayoutError:continue
        before=source_page.get_pixmap(matrix=fitz.Matrix(3,3),clip=box,alpha=False)
        after=edited_page.get_pixmap(matrix=fitz.Matrix(3,3),clip=box,alpha=False)
        a=np.frombuffer(before.samples,np.uint8).reshape(before.height,before.width,before.n)[:,:,:3].astype(float)
        b=np.frombuffer(after.samples,np.uint8).reshape(after.height,after.width,after.n)[:,:,:3].astype(float)
        color=np.array([(entry.color>>16)&255,(entry.color>>8)&255,entry.color&255])
        original=(np.max(np.abs(a-color),axis=2)<55)&(np.max(np.abs(a-bg),axis=2)>35)
        residual=(np.max(np.abs(b-color),axis=2)<55)&(np.max(np.abs(b-bg),axis=2)>35)
        if residual.sum()<12 or residual.sum()<original.sum()*.3:continue
        support=cv2.dilate(original.astype(np.uint8),np.ones((3,3),np.uint8))>0
        if (residual & support).sum()/residual.sum()<.9:continue
        edited_page.draw_rect(box,color=None,fill=(bg/255).tolist(),overlay=True)


def render(source: Path, destination: Path, placements: list[Placement], selected: list[int], layout: Layout, bilingual=True):
    css, archive = css_fonts(layout.fonts())
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Everything was preflighted before any source content is removed.
    with fitz.open(source) as original, fitz.open() as translated, fitz.open() as out:
        for index, number in enumerate(selected):
            # A fresh page copy isolates image replacement from all other pages.
            if len(translated):
                translated.delete_page(0)
            translated.insert_pdf(original, from_page=number-1, to_page=number-1)
            page = translated[0]
            entries = [x for x in placements if x.page == number]
            shadows = [fitz.Rect(b) for x in entries for b in (x.shadow_boxes or [])]
            for info in page.get_images(full=True):
                boxes = page.get_image_rects(info[0])
                if boxes and all(any(max(abs(x-y) for x,y in zip(b,s))<.01 for s in shadows) for b in boxes):
                    page.delete_image(info[0])
            native = [x for x in entries if x.native]
            owned_centers={(round((b[0]+b[2])/2,2),round((b[1]+b[3])/2,2)) for e in native for b in e.erase}
            foreign=[]
            for block in page.get_text('rawdict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        for char in span['chars']:
                            b=char['bbox'];center=(round((b[0]+b[2])/2,2),round((b[1]+b[3])/2,2))
                            if not char['c'].isspace() and center not in owned_centers:foreign.append(fitz.Rect(b))
            for entry in native:
                for box in entry.erase:
                    # Redaction removes the whole intersecting glyph. Using its
                    # full font box may also remove a nearby formula's parenthesis
                    # whose ascender overlaps the box without touching the ink.
                    b=fitz.Rect(box);chosen=None
                    nearby=[o for o in foreign if intersects(o,b,epsilon=0)]
                    for fx,fy in [(0.5,.5),(.5,.2),(.5,.8),(.2,.5),(.8,.5),(.2,.2),(.8,.2),(.2,.8),(.8,.8)]:
                        x=b.x0+b.width*fx;y=b.y0+b.height*fy;half=min(.12,b.width*.08,b.height*.08)
                        candidate=fitz.Rect(x-half,y-half,x+half,y+half)
                        if not any(intersects(candidate,o,epsilon=0) for o in nearby):chosen=candidate;break
                    if chosen is None:raise LayoutError(f'No safe glyph redaction point: {entry.id}',entry.id)
                    page.add_redact_annot(chosen, fill=None, cross_out=False)
            if native:
                # Native edits never touch image pixels or vector strokes.
                page.apply_redactions(images=0, graphics=0, text=0)
                clear_native_bitmap_replicas(original[number-1],page,native)
            # Restore all backgrounds before ANY translated glyphs. Overlapping
            # OCR boxes must not erase a translation inserted earlier in the loop.
            for entry in entries:
                if not entry.native and not entry.raster_patch:
                    page.draw_rect(fitz.Rect(entry.background_box or entry.source_bbox),color=None,fill=entry.background,overlay=True)
            patch_entries=[e for e in entries if e.raster_patch]
            clusters=[]
            for entry in patch_entries:
                cluster=[entry];box=fitz.Rect(entry.raster_patch_box)
                for previous in list(clusters):
                    if intersects(box,previous[0],epsilon=0):
                        box|=previous[0];cluster+=previous[1];clusters.remove(previous)
                clusters.append((box,cluster))
            from PIL import Image
            import io
            for box,cluster in clusters:
                pix=page.get_pixmap(matrix=fitz.Matrix(4,4),clip=box,alpha=False)
                composite=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
                for entry in cluster:
                    with Image.open(entry.raster_patch) as patch:
                        offset=(round(entry.raster_patch_box[0]*4)-pix.x,round(entry.raster_patch_box[1]*4)-pix.y)
                        composite.paste(patch,offset,patch.getchannel('A'))
                # Every overlapping crop sees the fully cleaned canvas, while
                # gaps between crops retain the original PDF artwork exactly.
                for entry in cluster:
                    actual=fitz.Rect(entry.raster_patch_box)
                    bounds=(round(actual.x0*4)-pix.x,round(actual.y0*4)-pix.y,
                            round(actual.x1*4)-pix.x,round(actual.y1*4)-pix.y)
                    data=io.BytesIO();composite.crop(bounds).save(data,format='PNG')
                    page.insert_image(actual,stream=data.getvalue(),overlay=True)
            for entry in entries:
                if not entry.text and not entry.html_body:continue  # erase-only local recovery
                for name, path in (entry.assets or {}).items():
                    archive.add((Path(path).read_bytes(), name))
                try:
                    spare, scale = page.insert_htmlbox(fitz.Rect(entry.frame), content_html(entry.text, entry.size, entry.color, entry.bold, entry.align, layout.line_height, entry.html_body),
                                                       css=css, archive=archive, scale_low=layout.min_font_scale,rotate=entry.rotate)
                except (RuntimeError,ValueError) as exc:
                    raise LayoutError(f'Cannot insert text for {entry.id}: {exc}',entry.id) from exc
                if spare < 0 or abs(scale-entry.scale) > 0.01:
                    raise LayoutError(f"Placement changed after preflight: {entry.id}",entry.id)
            if bilingual:
                out.insert_pdf(original, from_page=number-1, to_page=number-1)
            out.insert_pdf(translated, from_page=0, to_page=0)
        out.set_metadata({"title": source.stem + " - SlideTwin", "producer": "SlideTwin"})
        out.subset_fonts()
        out.save(destination, garbage=4, deflate=True)
