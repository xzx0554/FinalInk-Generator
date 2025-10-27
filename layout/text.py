from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from core import RenderCfg, RuntimeFlags
from engine import ops as E
import random
import math
import numpy as np

def draw_text_rgba(text: str, fonts: List[ImageFont.FreeTypeFont], rcfg: RenderCfg, seed: int):
    E.seed_all(seed); W,H,M = rcfg.width, rcfg.height, rcfg.margin
    canvas = Image.new('RGBA', (W,H), (0,0,0,0))
    A = E.ri(max(0, rcfg.baseline_amp_px[0]//2), max(1, rcfg.baseline_amp_px[1]//2))
    T = E.ri(*rcfg.baseline_period_px); phi = random.random()*2*math.pi
    xs = np.arange(W, dtype=np.float32); baseline = (A*np.sin(2*math.pi*xs/T + phi)).astype(np.int32)

    slant = E.ru(*rcfg.slant_range)
    lines = text.split('\n'); y_cursor = M; anchors = []
    illeg = RuntimeFlags.illegibility

    for line in lines:
        if not line:
            y_cursor += E.ri(int(rcfg.font_max*1.05), int(rcfg.font_max*1.12))
            continue
        x = M
        base_size = E.ri(rcfg.font_min, rcfg.font_max)
        decay = E.ru(*rcfg.per_char_scale_decay); alpha_decay = E.ru(*rcfg.fatigue_alpha_drop); alpha_mult = 1.0
        last_anchor = None

        for i,ch in enumerate(line):
            fsize = max(18, int(base_size * (decay ** i)))
            font = fonts[0] if fonts else None
            if font is None:
                font = random.choice(fonts)

            # 初始蒙版
            mask = Image.new('L', (fsize*3, fsize*3), 0)
            md = ImageDraw.Draw(mask)
            jitter_y = int(np.random.normal(0, fsize*max(0.02, rcfg.jitter_y_sigma_ratio*0.6)))
            md.text((fsize//2, fsize//2 + jitter_y), ch, font=font.font_variant(size=fsize), fill=255)

            # 小斜切 + 小旋转
            m1 = E.shear_affine_mask(mask, shx=slant*0.5 + np.random.normal(0,0.03))
            mask = m1.rotate(E.ru(rcfg.per_char_rot[0]*0.6, rcfg.per_char_rot[1]*0.6), resample=Image.BILINEAR, expand=True)
            mask = mask.point(lambda p: int(max(0, min(255, p*alpha_mult))))

            tile = Image.new('RGBA', mask.size, (0,0,0,0))
            tile.putalpha(mask)

            # —— 单字内部“多重破坏”流水线（强度随 illegibility 提升）——
            # 强弹性 + 向心塌陷 + 横向挤压
            amp_lo = 12.0 + 18.0*illeg; amp_hi = 18.0 + 24.0*illeg
            tile = E.glyph_elastic_inmask(tile, alpha_range=(amp_lo, amp_hi), sigma_range=(3.0, 5.0), seed=seed+i*31+1)
            k_lo = 0.28 + 0.22*illeg; k_hi = 0.40 + 0.30*illeg
            tile = E.glyph_radial_collapse(tile, k_range=(k_lo, k_hi), noise=0.5+0.2*illeg, seed=seed+i*31+2)
            tile = E.glyph_collapse_x(tile, amp_px=max(2.0, fsize*(0.06+0.04*illeg)), period_px=max(5, int(fsize*0.22)), seed=seed+i*31+3)

            # 叠写/拖墨可控（默认关闭，避免重影）
            if not RuntimeFlags.disable_overstrike and illeg > 0.5:
                tile = E.glyph_overstrike(tile, n=E.ri(1,2), max_rot=10, max_shx=0.22, alpha=E.ri(130,180), seed=seed+i*31+4)
            if not RuntimeFlags.disable_drag and illeg > 0.3:
                tile = E.glyph_ink_drag_x(tile, max_px=E.ri(1,3), steps=E.ri(1,2), seed=seed+i*31+5)

            # 微小缺笔 + 紧凑性守卫
            n1 = int(10 + 18*illeg)
            tile = E.glyph_micro_dropout(tile, n_range=(n1-3, n1+3), r_range=(1,3), erase_ratio=0.75, seed=seed+i*31+6)
            tile = E.enforce_compactness(tile, max_gap_ratio=0.40)

            gw, gh = tile.size

            # —— 自动换行（严格不外溢）——
            if x + gw > W - M:
                x = M
                y_cursor += int(base_size * E.ru(1.05, 1.12))
                last_anchor = None  

            by = y_cursor + int(baseline[min(W-1, max(0, x))])
            if by + gh > H - M:
                break

            canvas.alpha_composite(tile, dest=(x, by))

            # 连笔（提笔困难的感觉）：随不可辨度略升，但绝不跨行
            if random.random() < max(0.08, rcfg.connect_prob*0.4 + 0.15*illeg):
                bbox = mask.getbbox()
                if bbox:
                    left_x = x + 2; right_x = x + gw - 3; mid_y = by + gh//2 + E.ri(-2,2)
                    start_anchor = (left_x, mid_y); end_anchor = (right_x, mid_y)
                    if last_anchor is not None:
                        anchors.append((last_anchor, start_anchor))
                    last_anchor = end_anchor
            else:
                last_anchor = None

            # 前进（略紧凑）
            adv = int(fsize * E.ru(max(0.55, rcfg.char_spacing_ratio[0]), max(0.85, rcfg.char_spacing_ratio[1])))
            x += max(1, adv)
            alpha_mult *= alpha_decay

        y_cursor += int(base_size * E.ru(1.05, 1.10))

    return canvas, anchors
def connect_cursive(img_rgba: Image.Image, anchors: List[Tuple[Tuple[int,int],Tuple[int,int]]], rcfg: RenderCfg, seed: int) -> Image.Image:
    E.seed_all(seed); W,H=img_rgba.size; layer=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(layer)
    for (p0,p1) in anchors:
        if random.random()>rcfg.connect_prob: continue
        mid_cnt=E.ri(2,3); pts=[p0]
        for k in range(mid_cnt):
            t=(k+1)/(mid_cnt+1)
            x=int(p0[0]*(1-t)+p1[0]*t + np.random.normal(0,6))
            y=int(p0[1]*(1-t)+p1[1]*t + np.random.normal(0,6))
            pts.append((x,y))
        pts.append(p1)
        for i in range(len(pts)-1):
            d.line([pts[i],pts[i+1]], fill=(0,0,0,255), width=E.ri(*rcfg.connect_width_px))
    layer=layer.filter(ImageFilter.GaussianBlur(radius=E.ru(0.4,1.2))); img_rgba.alpha_composite(layer); return img_rgba