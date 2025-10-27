from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
@dataclass
class RenderCfg:
    width: int = 1024; height: int = 512; margin: int = 40
    font_min: int = 80; font_max: int = 120
    per_char_scale_decay: Tuple[float,float] = (0.97, 0.992)
    line_spacing_ratio: Tuple[float,float] = (0.9, 1.4)
    char_spacing_ratio: Tuple[float,float] = (0.4, 1.1)
    overlap_prob: float = 0.5; overlap_ratio: Tuple[float,float] = (0.10, 0.35)
    baseline_amp_px: Tuple[int,int] = (10, 18); baseline_period_px: Tuple[int,int] = (110, 220)
    jitter_y_sigma_ratio: float = 0.10; per_char_rot: Tuple[float,float] = (-6.0, 6.0)
    fatigue_alpha_drop: Tuple[float,float] = (0.975, 0.995)
    slant_range: Tuple[float,float] = (-0.35, 0.35)
    connect_prob: float = 0.65; connect_width_px: Tuple[int,int] = (1, 3)

@dataclass
class StylizeCfg:
    stroke_jitter_amp: Tuple[float,float] = (2.5, 5.0); stroke_jitter_period: Tuple[int,int] = (6, 12)
    elastic_alpha: Tuple[float,float] = (12.0, 20.0); elastic_sigma: Tuple[float,float] = (6.0, 10.0)
    pressure_range: Tuple[float,float] = (0.55, 1.7)
    dropout_len_px: Tuple[int,int] = (6, 18); dropout_prob: float = 0.03; curve_dropout_prob: float = 0.25
    motion_blur_ksize: Tuple[int,int] = (7, 13); motion_blur_jitter: Tuple[float,float] = (-10.0, 10.0)
    contour_rough_px: Tuple[float,float] = (0.8, 1.8); gamma_range: Tuple[float,float] = (0.80, 1.20)
    down_up_scale: Tuple[float,float] = (0.32, 0.68); multi_gauss_pass: Tuple[int,int] = (1, 3)
    gauss_sigma_range: Tuple[float,float] = (1.0, 2.6); alpha_erosion_px: Tuple[int,int] = (0, 2)
    mesh_grid: Tuple[int,int] = (8, 4); mesh_jitter: Tuple[float,float] = (-0.18, 0.18)

@dataclass
class Bucket: name: str; render: RenderCfg; style: StylizeCfg

@dataclass
class RuntimeFlags:
    illegibility: float = 0.8
    disable_overstrike: bool = True
    disable_drag: bool = True

def make_bucket(name: str) -> Bucket:
    if name == 'light':
        return Bucket('light',
            RenderCfg(per_char_scale_decay=(0.985,0.997), overlap_prob=0.25, jitter_y_sigma_ratio=0.06,
                      baseline_amp_px=(4,8), per_char_rot=(-3,3), slant_range=(-0.18,0.18), connect_prob=0.35),
            StylizeCfg(stroke_jitter_amp=(1.0,2.0), stroke_jitter_period=(10,16), elastic_alpha=(5,8), elastic_sigma=(4,6),
                       pressure_range=(0.9,1.2), dropout_len_px=(0,6), dropout_prob=0.008, curve_dropout_prob=0.1,
                       motion_blur_ksize=(3,7), motion_blur_jitter=(-6,6), contour_rough_px=(0.4,1.0),
                       gamma_range=(0.9,1.1), down_up_scale=(0.6,0.85), multi_gauss_pass=(0,1),
                       gauss_sigma_range=(0.6,1.4), alpha_erosion_px=(0,1), mesh_grid=(6,3), mesh_jitter=(-0.10,0.10)))
    if name == 'heavy':
        return Bucket('heavy',
            RenderCfg(per_char_scale_decay=(0.97,0.992), overlap_prob=0.55, jitter_y_sigma_ratio=0.10,
                      baseline_amp_px=(10,16), per_char_rot=(-6,6), slant_range=(-0.35,0.35), connect_prob=0.7),
            StylizeCfg(stroke_jitter_amp=(2.5,5.0), stroke_jitter_period=(6,12), elastic_alpha=(12,20), elastic_sigma=(6,10),
                       pressure_range=(0.55,1.7), dropout_len_px=(6,18), dropout_prob=0.03, curve_dropout_prob=0.25,
                       motion_blur_ksize=(7,13), motion_blur_jitter=(-10,10), contour_rough_px=(0.8,1.8),
                       gamma_range=(0.8,1.2), down_up_scale=(0.32,0.68), multi_gauss_pass=(1,3),
                       gauss_sigma_range=(1.0,2.6), alpha_erosion_px=(0,2), mesh_grid=(8,4), mesh_jitter=(-0.18,0.18)))
    return Bucket('medium', RenderCfg(), StylizeCfg())

def make_extreme(bucket: Bucket) -> Bucket:
    bucket.name='extreme'; bucket.render.overlap_prob=0.75; bucket.render.overlap_ratio=(0.18,0.42)
    bucket.render.jitter_y_sigma_ratio=0.12; bucket.render.per_char_rot=(-9,9)
    bucket.render.per_char_scale_decay=(0.96,0.988); bucket.render.connect_prob=0.85
    bucket.style = StylizeCfg(stroke_jitter_amp=(3.0,6.0), stroke_jitter_period=(5,10), elastic_alpha=(16,26), elastic_sigma=(6,10),
                              pressure_range=(0.5,1.8), dropout_len_px=(8,24), dropout_prob=0.05, curve_dropout_prob=0.35,
                              motion_blur_ksize=(9,15), motion_blur_jitter=(-12,12), contour_rough_px=(1.2,2.2),
                              gamma_range=(0.75,1.25), down_up_scale=(0.28,0.58), multi_gauss_pass=(2,4),
                              gauss_sigma_range=(1.2,3.0), alpha_erosion_px=(1,3), mesh_grid=(10,5), mesh_jitter=(-0.22,0.22))
    return bucket
