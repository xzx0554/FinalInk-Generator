import argparse, os, random
from core import make_bucket, make_extreme, RuntimeFlags
from pipeline import apply_readability, process_once
from engine.store import load_fonts, save_sample
from engine.store import ensure_dir
from engine import ops as E

def parse_args():
    ap=argparse.ArgumentParser(description="FinalInk Generator is not just an artistic tool—it is an assistive algorithm inspired by the moment when a person, near the end of life, may lose the ability to speak and can only communicate by handwriting.")
    ap.add_argument("--texts", nargs="*", help="直接在命令行提供的文本（可以多个）")
    ap.add_argument("--text_file", type=str, default=None, help="包含多行文本的文件路径（UTF-8）")
    ap.add_argument("--fonts", nargs="+", required=True, help="一个或多个中文字体路径（TTF/OTF）")
    ap.add_argument("--out", type=str, required=True, help="输出目录")
    ap.add_argument("--n_per_text", type=int, default=50, help="每条文本生成多少张")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--bucket", type=str, default="heavy", choices=["light","medium","heavy","extreme"], help="强度档位")
    ap.add_argument("--canvas_w", type=int, default=1024)
    ap.add_argument("--canvas_h", type=int, default=512)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--readability", type=float, default=0.35,
                help="可辨度 0~1（0=原强度，1=更清晰；建议 0.3~0.5）")
    return ap.parse_args()

def main():
    args=parse_args()
    corpus=[]; 
    if args.texts: corpus.extend(args.texts)
    if args.text_file and os.path.isfile(args.text_file):
        with open(args.text_file,"r",encoding="utf-8") as f:
            for line in f:
                t=line.strip("\r\n"); 
                if t: corpus.append(t)
    if not corpus: raise RuntimeError("没有可用文本。请使用 --texts 或 --text_file 提供待渲染文本。")
    bucket= make_bucket('heavy' if args.bucket=='extreme' else args.bucket)
    if args.bucket=='extreme': bucket=make_extreme(bucket)
    apply_readability(bucket, args.readability)
    bucket.render.width=args.canvas_w; bucket.render.height=args.canvas_h; bucket.render.margin=args.margin
    #注意这个可以在core里改，但是我懒得动了
    bucket.style.alpha_erosion_px = (0, 0)
    bucket.render.baseline_amp_px = (0, 2)
    bucket.render.per_char_rot = (-1.2, 1.2)
    bucket.render.jitter_y_sigma_ratio = 0.025
    bucket.render.per_char_scale_decay = (0.994, 0.997)
    bucket.render.slant_range = (-0.08, 0.08)
    bucket.render.char_spacing_ratio = (0.55, 0.85)   
    bucket.render.overlap_prob = 0.15                  
    bucket.render.connect_prob = 0.10                  
    bucket.render.line_spacing_ratio = (1.05, 1.12)   
    bucket.render.baseline_amp_px = (0, 2)
    bucket.render.per_char_rot = (-1.2, 1.2)
    bucket.render.jitter_y_sigma_ratio = 0.025
    bucket.render.per_char_scale_decay = (0.994, 0.997)
    bucket.render.slant_range = (-0.08, 0.08)
    bucket.render.char_spacing_ratio = (0.55, 0.85)
    bucket.render.overlap_prob = 0.12
    bucket.render.connect_prob = 0.10
    bucket.render.line_spacing_ratio = (1.05, 1.12)
    bucket.style.dropout_prob = 0.0
    bucket.style.curve_dropout_prob = 0.0
    bucket.style.alpha_erosion_px = (0, 0)
    bucket.style.stroke_jitter_amp = (1.0, 2.0)
    bucket.style.dropout_prob = 0.0
    bucket.style.curve_dropout_prob = 0.0
    bucket.style.alpha_erosion_px = (0, 0)

    fonts=load_fonts(args.fonts, size=bucket.render.font_max)
    ensure_dir(args.out); E.seed_all(args.seed)
    idx=0
    for t in corpus:
        for _ in range(max(1,args.n_per_text)):
            s=random.randrange(1<<30); img,meta=process_once(t, fonts, bucket, seed=s)
            save_sample(args.out, idx, img, meta); idx+=1
    print(f"Done. Generated {idx} samples into: {args.out}")