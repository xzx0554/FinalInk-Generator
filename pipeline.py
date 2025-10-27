from core import Bucket, RuntimeFlags
from layout.text import draw_text_rgba, connect_cursive
from engine import ops as E
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple, Dict
import numpy as np
import random
from dataclasses import dataclass, asdict

def lerp(a, b, t):
    return a + (b - a) * t

def clamp01(x): return max(0.0, min(1.0, x))

def alpha_gamma(img_rgba: Image.Image, gamma: float) -> Image.Image:
    r,g,b,a=img_rgba.split(); lut=[int((i/255.0)**gamma*255.0+0.5) for i in range(256)]; a2=a.point(lut*1); return Image.merge('RGBA',(r,g,b,a2))

def elastic_warp(img_rgba: Image.Image, alpha: float, sigma: float, seed: int) -> Image.Image:
    E.seed_all(seed); arr = np.array(img_rgba, np.uint8); H,W = arr.shape[0], arr.shape[1]
    scale = max(8, int(min(H,W)/48))
    fx = (E.rand_lowres_mask((H,W), scale=scale)*2-1) * alpha
    fy = (E.rand_lowres_mask((H,W), scale=scale)*2-1) * alpha
    def blur_f(m):
        im = Image.fromarray(((m - m.min())/(np.ptp(m)+1e-6)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(0.1, sigma)))
        return (np.array(im).astype(np.float32)/255.0)*(m.max()-m.min()) + m.min()

    fx = blur_f(fx); fy = blur_f(fy)
    yy,xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mx = np.clip((xx+fx).round().astype(np.int32), 0, W-1); my = np.clip((yy+fy).round().astype(np.int32), 0, H-1)
    warped = arr[my, mx, :]
    return Image.fromarray(warped, 'RGBA')

def pressure_variation(img_rgba: Image.Image, pr_min: float, pr_max: float, seed: int) -> Image.Image:
    E.seed_all(seed); r,g,b,a = img_rgba.split(); a_np = np.array(a, np.uint8); im = Image.fromarray(a_np,'L')
    for _ in range(E.ri(1,2)):
        im = im.filter(random.choice([ImageFilter.MaxFilter(size=3), ImageFilter.MinFilter(size=3), ImageFilter.MaxFilter(size=5), ImageFilter.MinFilter(size=5)]))
    mask = E.rand_lowres_mask(a_np.shape, scale=32)
    pr = E.ru(pr_min, pr_max); w = clamp01(0.3 + 0.7*(pr-pr_min)/(pr_max-pr_min+1e-6))
    mask = (mask*0.5 + w*0.5).astype(np.float32)
    out_a = (mask*np.array(im,np.float32) + (1-mask)*a_np.astype(np.float32)).astype(np.uint8)
    return Image.merge('RGBA', (r,g,b, Image.fromarray(out_a,'L')))

def apply_readability(bucket: Bucket, r: float) -> None:
    """
    r: 0~1，值越大越清晰。通过缩小扭曲/噪声/模糊、减少重合与连笔概率来提高可辨度。
    """
    r = float(max(0.0, min(1.0, r)))
    intensity = 1.0 - r

    rcfg, scfg = bucket.render, bucket.style

    # —— 布局/几何（更稳一点） ——
    rcfg.overlap_prob *= intensity
    rcfg.overlap_ratio = (rcfg.overlap_ratio[0]*intensity*0.75,
                          rcfg.overlap_ratio[1]*intensity*0.75)
    rcfg.jitter_y_sigma_ratio *= lerp(1.0, 0.6, r)
    rot_max = max(abs(rcfg.per_char_rot[0]), abs(rcfg.per_char_rot[1]))
    rot_new = rot_max * intensity * 0.6
    rcfg.per_char_rot = (-rot_new, rot_new)
    # 疲劳缩小更温和
    rcfg.per_char_scale_decay = (lerp(rcfg.per_char_scale_decay[0], 0.995, r),
                                 lerp(rcfg.per_char_scale_decay[1], 0.998, r))
    # 连笔概率下调，但不完全去掉
    rcfg.connect_prob = lerp(rcfg.connect_prob, 0.35, r)

    # —— 形变/噪声（弱化幅度与频率） ——
    scfg.elastic_alpha = (max(2.0, scfg.elastic_alpha[0]*intensity),
                          max(3.0, scfg.elastic_alpha[1]*intensity))
    scfg.mesh_jitter = (scfg.mesh_jitter[0]*intensity*0.7,
                        scfg.mesh_jitter[1]*intensity*0.7)
    scfg.stroke_jitter_amp = (scfg.stroke_jitter_amp[0]*intensity*0.7,
                              scfg.stroke_jitter_amp[1]*intensity*0.7)
    scfg.dropout_prob *= intensity*0.6
    scfg.curve_dropout_prob *= intensity*0.6
    scfg.dropout_len_px = (int(scfg.dropout_len_px[0]*intensity),
                           int(max(2, scfg.dropout_len_px[1]*intensity)))
    scfg.contour_rough_px = (scfg.contour_rough_px[0]*intensity*0.6,
                             scfg.contour_rough_px[1]*intensity*0.6)
    # 腐蚀更少
    scfg.alpha_erosion_px = (0, max(0, int(round(scfg.alpha_erosion_px[1]*intensity*0.5))))

    # —— 模糊（减小核/次数/角度抖动） ——
    kmin, kmax = scfg.motion_blur_ksize
    kmin = max(3, int(round(kmin * lerp(1.0, 0.6, r))))
    kmax = max(kmin, int(round(kmax * lerp(1.0, 0.6, r))))
    if kmin % 2 == 0: kmin += 1
    if kmax % 2 == 0: kmax += 1
    scfg.motion_blur_ksize = (kmin, kmax)
    scfg.motion_blur_jitter = (scfg.motion_blur_jitter[0]*intensity,
                               scfg.motion_blur_jitter[1]*intensity)

    # 强降采样糊度 → 更接近 1（更清晰），高斯次数减少
    scfg.down_up_scale = (lerp(scfg.down_up_scale[0], 0.75, r),
                          lerp(scfg.down_up_scale[1], 0.90, r))
    scfg.multi_gauss_pass = (0, max(0, int(round(scfg.multi_gauss_pass[1]*intensity))))
    scfg.gauss_sigma_range = (scfg.gauss_sigma_range[0]*intensity,
                              scfg.gauss_sigma_range[1]*intensity)
    # gamma 收敛到 1
    scfg.gamma_range = (lerp(scfg.gamma_range[0], 0.95, r),
                        lerp(scfg.gamma_range[1], 1.05, r))
    
    scfg = bucket.style

    # 禁掉运动模糊
    scfg.motion_blur_ksize = (1, 1)
    scfg.motion_blur_jitter = (0.0, 0.0)

    # 禁掉降采样-升采样中的高斯（虽然函数里已不再使用，但这里也给出“意图上的”约束）
    scfg.multi_gauss_pass = (0, 0)
    scfg.gauss_sigma_range = (0.0, 0.0)

    # 降采样比例更保守一点（仍可制造难看但不糊的锯齿）
    scfg.down_up_scale = (max(0.5, scfg.down_up_scale[0]), min(0.9, scfg.down_up_scale[1]))
    
def process_once(text: str, fonts: List[ImageFont.FreeTypeFont], bucket: Bucket, seed: int):
    rcfg, scfg = bucket.render, bucket.style
    img, anchors = draw_text_rgba(text, fonts, rcfg, seed=seed)
    img = elastic_warp(img, alpha=E.ru(2.0, 4.0), sigma=E.ru(2.5, 3.5), seed=seed+3)
    # 压力/轻裁剪，但不损坏连通性
    img = pressure_variation(img, pr_min=max(0.8, scfg.pressure_range[0]), pr_max=min(1.5, scfg.pressure_range[1]), seed=seed+4)
    # 少量连笔（draw_text_rgba里已做 anchors），这里不再添加新的
    img = connect_cursive(img, anchors, rcfg, seed=seed+5)  # 如需更多连笔可打开
    # 不再做腐蚀；gamma 收敛到近1
    # img = alpha_erosion(img, px=0)
    img = alpha_gamma(img, gamma=E.ru(0.95, 1.05))
    meta = {"text": text, "bucket": bucket.name, "seed": seed, "render": asdict(bucket.render), "style": asdict(bucket.style)}
    return img, meta