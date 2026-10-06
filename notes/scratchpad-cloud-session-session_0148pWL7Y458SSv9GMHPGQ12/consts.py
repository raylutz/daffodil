import re
try:
    import cv2
    print("opencv", cv2.__version__)
    for n in ['IMREAD_COLOR','IMREAD_GRAYSCALE','IMREAD_UNCHANGED','THRESH_BINARY','THRESH_OTSU','CALIB_FIX_ASPECT_RATIO']:
        v=getattr(cv2,n); print(f"  cv2.{n:24} = {v!r:6} type {type(v).__name__}")
    print("  combined:", cv2.THRESH_BINARY | cv2.THRESH_OTSU, "type", type(cv2.THRESH_BINARY | cv2.THRESH_OTSU).__name__)
except Exception as e:
    print("opencv not available:", type(e).__name__, e)
