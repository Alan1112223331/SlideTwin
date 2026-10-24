"""Source bitmap geometry shared by extraction and layout."""
import math

import numpy as np
import pymupdf as fitz


def visible_clip(page: fitz.Page, box) -> fitz.Rect | None:
    """Return the nonempty part of an object inside the visible PDF page."""
    try:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            return None
        box = fitz.Rect(box)
    except (TypeError, ValueError, OverflowError):
        # Geometry also comes from cached extraction records. An invalid box
        # must not abort the page while selecting its visible objects.
        return None
    if box.is_empty or box.is_infinite:
        return None
    clip = box & page.rect
    return None if clip.is_empty else clip


def clipped_pixmap(page: fitz.Page, box, *, scale=1, padding=0) -> fitz.Pixmap | None:
    """Rasterize visible source content without sending empty images downstream.

    Check the original box before padding: padding must not turn an off-page
    object into an analysis of unrelated artwork at the page edge. Also check
    pixel dimensions because a valid floating-point clip can rasterize empty.
    """
    clip = visible_clip(page, box)
    if clip is None:
        return None
    if padding:
        clip = (clip + (-padding, -padding, padding, padding)) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    if pix.width <= 0 or pix.height <= 0 or not pix.samples:
        return None
    return pix


def raster_container(page: fitz.Page, box: fitz.Rect, bridge_glyphs=False) -> fitz.Rect | None:
    """Find a flat bitmap cell bounded by its actual colored outline."""
    clip = visible_clip(page, box)
    if clip is None:
        return None
    import cv2
    scale=2
    # Reuse one raster for immutable source-page geometry; hundreds of labels
    # on a slide must not rasterize the entire slide hundreds of times.
    content_key=tuple(page.get_contents())
    cache=getattr(page,'_slidetwin_cell_raster',None)
    if cache is None or cache[0]!=content_key:
        pix=clipped_pixmap(page,page.rect,scale=scale)
        if pix is None:return None
        a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
        cache=(content_key,pix,a,{})
        page._slidetwin_cell_raster=cache
    _,pix,a,contour_cache=cache
    crop=a[max(0,int(clip.y0*scale)):int(clip.y1*scale)+1,max(0,int(clip.x0*scale)):int(clip.x1*scale)+1]
    if not crop.size:return None
    colors,counts=np.unique(crop.reshape(-1,3)//12,axis=0,return_counts=True)
    bucket=colors[counts.argmax()];pixels=crop.reshape(-1,3)
    bg=np.median(pixels[np.all(pixels//12==bucket,axis=1)],axis=0)
    color_key=(*bg.tolist(),bridge_glyphs)
    if color_key not in contour_cache:
        mask=(np.max(np.abs(a.astype(float)-bg),axis=2)<12).astype(np.uint8)
        if bridge_glyphs:mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contour_cache[color_key]=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours=contour_cache[color_key]
    found=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);rect=fitz.Rect(x/scale,y/scale,(x+w)/scale,(y+h)/scale)
        if (box.tl+box.br)/2 not in rect or (rect & box).get_area()<box.get_area()*.8:continue
        if w*h>pix.width*pix.height*.35 or w<8 or h<8:continue
        if cv2.contourArea(contour)/(w*h)<.8:continue
        found.append(rect)
    return min(found,key=lambda r:r.get_area()) if found else None

