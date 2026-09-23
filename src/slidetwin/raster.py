"""Source bitmap geometry shared by extraction and layout."""
import numpy as np
import pymupdf as fitz


def raster_container(page: fitz.Page, box: fitz.Rect, bridge_glyphs=False) -> fitz.Rect | None:
    """Find a flat bitmap cell bounded by its actual colored outline."""
    import cv2
    scale=2
    # Reuse one raster for immutable source-page geometry; hundreds of labels
    # on a slide must not rasterize the entire slide hundreds of times.
    content_key=tuple(page.get_contents())
    cache=getattr(page,'_slidetwin_cell_raster',None)
    if cache is None or cache[0]!=content_key:
        pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
        a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
        cache=(content_key,pix,a,{})
        page._slidetwin_cell_raster=cache
    _,pix,a,contour_cache=cache
    crop=a[max(0,int(box.y0*scale)):int(box.y1*scale)+1,max(0,int(box.x0*scale)):int(box.x1*scale)+1]
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

