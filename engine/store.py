from PIL import ImageFont
import json, os
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFilter
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def load_fonts(paths: List[str], size: int) -> List[ImageFont.FreeTypeFont]:
    fonts=[]
    for p in paths:
        try:
            ft=ImageFont.truetype(p, size=size); fonts.append(ft)
        except Exception as e: print(f"[WARN] 字体加载失败：{p} -> {e}")
    if not fonts: raise RuntimeError("未能加载任何字体，请检查 --fonts 路径（须支持中文）。")
    return fonts

def save_sample(out_dir: str, idx: int, img: Image.Image, meta: Dict):
    ensure_dir(out_dir); stem=os.path.join(out_dir, f"sample_{idx:06d}")
    img.save(stem+".png"); 
    with open(stem+".json","w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
