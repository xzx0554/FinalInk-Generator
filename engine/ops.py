from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, random, math
from typing import List, Tuple, Dict

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed & 0xffffffff)
    
def ru(a,b): return random.random()*(b-a)+a

def ri(a,b): return random.randint(a,b)

def rand_lowres_mask(shape: Tuple[int,int], scale: int = 32, smooth_radius: float = 1.2) -> np.ndarray:
    H, W = shape; h, w = max(1, H // scale), max(1, W // scale)
    small = np.random.rand(h, w).astype(np.float32)
    big = Image.fromarray((small*255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    if smooth_radius>0: big = big.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
    return np.array(big, dtype=np.float32)/255.0

def shear_affine(img: Image.Image, shx: float) -> Image.Image:
    w,h = img.size
    m = (1, shx, 0, 0, 1, 0)
    pad = int(abs(shx)*h)
    canvas = Image.new('RGBA', (w+pad*2, h), (0,0,0,0))
    canvas.paste(img, (pad,0))
    return canvas.transform(canvas.size, Image.AFFINE, m, resample=Image.NEAREST)

def _alpha_weight(a_np: np.ndarray, edge_soften_px: int = 2) -> np.ndarray:
    """把字的 alpha 变成 0~1 的权重，并做一点边缘软化，避免边缘撕裂。"""
    w = (a_np.astype(np.float32)/255.0)
    if edge_soften_px > 0:
        im = Image.fromarray((w*255).astype(np.uint8), 'L').filter(ImageFilter.GaussianBlur(edge_soften_px))
        w = np.array(im, np.float32)/255.0
    return np.clip(w, 0.0, 1.0)

def glyph_elastic_inmask(tile_rgba: Image.Image, alpha_range: Tuple[float,float], sigma_range: Tuple[float,float], seed: int) -> Image.Image:
    """只在字形alpha区域内做强弹性形变；形变被 alpha 权重调制，不会把部件拉远。"""
    seed_all(seed)
    r,g,b,a = tile_rgba.split()
    a_np = np.array(a, np.uint8)
    H,W = a_np.shape
    if H < 4 or W < 4: return tile_rgba

    wmask = _alpha_weight(a_np, edge_soften_px=2)  # 0~1
    amp  = ru(*alpha_range)  # 形变强度，比全局 elastic 大
    sig  = ru(*sigma_range)
    scale = max(6, int(min(H,W)/40))
    fx = (rand_lowres_mask((H,W), scale=scale)*2-1) * amp * wmask
    fy = (rand_lowres_mask((H,W), scale=scale)*2-1) * amp * wmask

    def blur_f(m):
        im = Image.fromarray(((m - m.min())/(np.ptp(m)+1e-6)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(0.8, sig)))
        return (np.array(im).astype(np.float32)/255.0)*(m.max()-m.min()) + m.min()

    fx = blur_f(fx); fy = blur_f(fy)

    yy,xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mx = np.clip((xx+fx).round().astype(np.int32), 0, W-1)
    my = np.clip((yy+fy).round().astype(np.int32), 0, H-1)

    arr = np.dstack([np.array(ch, np.uint8) for ch in (r,g,b,a)])
    warped = arr[my, mx, :]
    return Image.fromarray(warped, 'RGBA')

def glyph_radial_collapse(tile_rgba: Image.Image, k_range=(0.25,0.55), noise=0.5, seed: int = 0) -> Image.Image:
    # 径向向心塌陷：把细节向质心拉，笔画互相挤；仍在本字内
    seed_all(seed)
    arr = np.array(tile_rgba, np.uint8); H,W = arr.shape[:2]
    A = arr[...,3]; ys,xs = np.where(A>60)
    if len(xs)==0: return tile_rgba
    cx, cy = float(xs.mean()), float(ys.mean())
    yy,xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx, dy = cx-xx, cy-yy
    dist = np.sqrt(dx*dx+dy*dy)+1e-6
    k = ru(*k_range)  # 强度
    n = (rand_lowres_mask((H,W), scale=max(8,int(min(H,W)/36)))*2-1)*noise
    mx = np.clip((xx + dx/dist * k*dist/(dist.max()+1e-6) + n).round().astype(np.int32), 0, W-1)
    my = np.clip((yy + dy/dist * k*dist/(dist.max()+1e-6) + n).round().astype(np.int32), 0, H-1)
    return Image.fromarray(arr[my, mx, :], 'RGBA')

def glyph_collapse_x(tile_rgba: Image.Image, amp_px: float, period_px: float, seed: int) -> Image.Image:
    """在字内按行做横向挤压/错位，幅度受 alpha 权重调制，模拟用力抖手但不撕裂。"""
    seed_all(seed)
    arr = np.array(tile_rgba, np.uint8)
    H,W = arr.shape[0], arr.shape[1]
    A = arr[...,3]
    wmask = _alpha_weight(A, edge_soften_px=1)
    out = np.zeros_like(arr)
    phase = random.random()*2*math.pi
    for y in range(H):
        # 字内强，背景弱
        local_amp = amp_px * float(np.mean(wmask[y:y+1, :]))
        shift = int(local_amp * math.sin(2*math.pi*y/max(3,period_px) + phase + random.random()*0.3))
        if shift >= 0:
            out[y, shift:, :] = arr[y, :W-shift, :]
        else:
            out[y, :W+shift, :] = arr[y, -shift:, :]
    return Image.fromarray(out, 'RGBA')

def glyph_overstrike(tile_rgba: Image.Image, n=2, max_rot=12, max_shx=0.25, alpha=170, seed: int = 0) -> Image.Image:
    # 叠写：同一个字重复盖写n次（轻微旋转/斜切/缩放），临终“重描”感
    seed_all(seed)
    base = tile_rgba.copy()
    W,H = base.size
    for i in range(n):
        t = tile_rgba.copy()
        ang = random.uniform(-max_rot, max_rot)
        shx = random.uniform(-max_shx, max_shx)
        s = random.uniform(0.9, 1.1)
        t = t.resize((max(2,int(W*s)), max(2,int(H*s))), Image.BILINEAR)
        r,g,b,a = t.split()
        a = a.point(lambda p: min(255, int(p*alpha/255)))
        t = Image.merge('RGBA',(r,g,b,a))
        t = shear_affine(t, shx)
        t = t.rotate(ang, resample=Image.BILINEAR, expand=True)
        # 居中粘回
        x = (W - t.size[0])//2; y = (H - t.size[1])//2
        base.alpha_composite(t, dest=(x,y))
    return base

def glyph_ink_drag_x(tile_rgba: Image.Image, max_px=6, steps=5, seed: int = 0) -> Image.Image:
    # 单向拖墨（沿书写方向x+），近似“用力拖笔”
    seed_all(seed)
    r,g,b,a = tile_rgba.split()
    a_np = np.array(a, np.float32)
    out = np.zeros_like(a_np)
    for i in range(steps):
        dx = ri(1, max_px)
        w  = 1.0 / (i+1)
        shifted = Image.fromarray(a_np.astype(np.uint8),'L').transform(
            a.size, Image.AFFINE, (1,0,dx, 0,1,0), resample=Image.BILINEAR)
        out = np.maximum(out, np.array(shifted, np.float32)*w)
    a2 = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'L')
    return Image.merge('RGBA', (r,g,b,a2))

def shear_affine_mask(mask: Image.Image, shx: float) -> Image.Image:
    # 输入为 L 蒙版
    w,h = mask.size
    m = (1, shx, 0, 0, 1, 0)
    pad = int(abs(shx)*h)
    canvas = Image.new('L', (w+pad*2, h), 0)
    canvas.paste(mask, (pad,0))
    return canvas.transform(canvas.size, Image.AFFINE, m, resample=Image.BILINEAR)

def glyph_micro_dropout(tile_rgba: Image.Image, n_range=(10,20), r_range=(1,3), erase_ratio=0.7, seed: int = 0) -> Image.Image:
    seed_all(seed)
    W,H = tile_rgba.size
    arr = np.array(tile_rgba, np.uint8)
    A = arr[...,3]
    ys,xs = np.where(A>60)
    if len(xs) == 0: return tile_rgba
    layer = Image.new('L', (W,H), 0)
    d = ImageDraw.Draw(layer)
    n = ri(*n_range)
    for _ in range(n):
        idx = ri(0, len(xs)-1); cx, cy = int(xs[idx]), int(ys[idx])
        rx, ry = ri(*r_range), ri(*r_range)
        if random.random() < erase_ratio:
            d.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)
        else:
            d.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=ri(120,180))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.8))
    L = np.array(layer, np.uint8)
    A = np.clip(A.astype(np.int16) - (L.astype(np.int16)*ri(150,220)//255), 0, 255).astype(np.uint8)
    arr[...,3] = A
    return Image.fromarray(arr, 'RGBA')

def enforce_compactness(tile_rgba: Image.Image, max_gap_ratio: float = 0.42) -> Image.Image:
    # 连通域之间距离过大就做一次 closing，避免“偏旁分家”
    arr = np.array(tile_rgba, np.uint8); A = arr[...,3]; H,W = A.shape
    th = (A > 80).astype(np.uint8); visited = np.zeros_like(th, bool); comps=[]
    for y in range(H):
        for x in range(W):
            if th[y,x] and not visited[y,x]:
                stack=[(y,x)]; visited[y,x]=True; ys=[]; xs=[]
                while stack:
                    cy,cx=stack.pop(); ys.append(cy); xs.append(cx)
                    for ny,nx in((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0<=ny<H and 0<=nx<W and th[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx]=True; stack.append((ny,nx))
                comps.append((np.array(ys),np.array(xs)))
    if len(comps)<=1: return tile_rgba
    comps.sort(key=lambda c: c[0].size, reverse=True)
    ys1,xs1 = comps[0]; ys2,xs2 = comps[1]
    c1 = (xs1.mean(), ys1.mean()); c2 = (xs2.mean(), ys2.mean())
    gap = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
    if gap > max_gap_ratio * max(W,H):
        A2 = Image.fromarray(A,'L').filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.MinFilter(size=3))
        arr[...,3] = np.array(A2, np.uint8)
    return Image.fromarray(arr, 'RGBA')


# 以下为废弃方法，但是完全可以使用与上面方法相同方法调用
# img = stroke_jitter_x(img, amp_px=ru(*scfg.stroke_jitter_amp), period_px=ru(*scfg.stroke_jitter_period), seed=seed+1)
# img = mesh_warp(img, grid_xy=scfg.mesh_grid, jitter=scfg.mesh_jitter, seed=seed+2)
# img = ink_blobs_and_holes(img, seed=seed+10)

#书写效果
def stroke_jitter_x(img_rgba: Image.Image, amp_px: float, period_px: float, seed: int) -> Image.Image:
    seed_all(seed); arr = np.array(img_rgba, np.uint8); H,W = arr.shape[0], arr.shape[1]
    out = np.zeros_like(arr); phase = random.random()*2*math.pi
    for y in range(H):
        shift = int(amp_px * math.sin(2*math.pi*y/period_px + phase + random.random()*0.2))
        if shift>=0: out[y, shift:, :] = arr[y, :W-shift, :]
        else:       out[y, :W+shift, :] = arr[y, -shift:, :]
    return Image.fromarray(out, 'RGBA')
def ink_blobs_and_holes(img_rgba: Image.Image, seed: int) -> Image.Image:
    seed_all(seed); W,H=img_rgba.size; arr=np.array(img_rgba,np.uint8); A=arr[...,3]; ys,xs=np.where(A>80)
    if len(xs)==0: return img_rgba
    layer=Image.new('L',(W,H),0); d=ImageDraw.Draw(layer)
    for _ in range(ri(10,25)):
        idx=ri(0,len(xs)-1); cx,cy=int(xs[idx]),int(ys[idx]); rx,ry=ri(1,4),ri(1,4)
        if random.random()<0.5: d.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], fill=180+ri(40,75))
        else: d.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], fill=255)
    layer=layer.filter(ImageFilter.GaussianBlur(radius=1.0)); L=np.array(layer,np.uint8)
    add_mask=(L>100)&(L<200); erase_mask=(L>=200)
    A=A.astype(np.int16); A[add_mask]=np.clip(A[add_mask]+40,0,255); A[erase_mask]=np.clip(A[erase_mask]-200,0,255)
    arr[...,3]=A.astype(np.uint8); return Image.fromarray(arr,'RGBA')
def mesh_warp(img_rgba: Image.Image, grid_xy: Tuple[int,int], jitter: Tuple[float,float], seed: int) -> Image.Image:
    seed_all(seed)
    W, H = img_rgba.size
    gx, gy = grid_xy
    cw, ch = W // gx, H // gy
    jx_scale = max(0.0, ru(*jitter)) * cw
    jy_scale = max(0.0, ru(*jitter)) * ch
    # Attension:这个方法非常保守，如果随机顶点的话，可以让字辨别读几乎为0，但是其不符合人类书写习惯，人一般不会将字不同的部分随机书写在纸面上
    verts = []
    for j in range(gy + 1):
        row = []
        for i in range(gx + 1):
            x = W if i == gx else i * cw
            y = H if j == gy else j * ch
            if 0 < i < gx:
                x += ri(-int(jx_scale), int(jx_scale))
            if 0 < j < gy:
                y += ri(-int(jy_scale), int(jy_scale))
            row.append((x, y))
        verts.append(row)

    mesh = []
    for j in range(gy):
        for i in range(gx):
            x0, y0 = i * cw, j * ch
            x1 = W if i == gx - 1 else (i + 1) * cw
            y1 = H if j == gy - 1 else (j + 1) * ch
            quad = [
                *verts[j][i],      
                *verts[j][i+1], 
                *verts[j+1][i+1], 
                *verts[j+1][i],
            ]
            mesh.append(((x0, y0, x1, y1), quad))

    return img_rgba.transform(img_rgba.size, Image.MESH, mesh, resample=Image.NEAREST)
# 纸面效果
def line_kernel_2d(ksize: int, angle_deg: float) -> np.ndarray:
    ksize = int(max(3, ksize))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cx = cy = ksize // 2
    ang = math.radians(angle_deg % 180.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for t in np.linspace(-ksize//2, ksize//2, ksize*4):
        x = int(round(cx + t * ca)); y = int(round(cy + t * sa))
        if 0 <= x < ksize and 0 <= y < ksize: kernel[y, x] = 1.0
    s = kernel.sum(); 
    if s>0: kernel /= s
    return kernel

def convolve_gray(img: Image.Image, kernel: np.ndarray) -> Image.Image:
    k = np.nan_to_num(kernel.astype(np.float32))
    h, w = k.shape
    # Pillow 要求卷积核 <=31x31，否则会报 bad kernel size
    if h > 31 or w > 31:
        h = w = 31
        k = np.ones((h, w), np.float32)
    s = float(k.sum()) if float(k.sum()) != 0 else 1.0
    k /= s
    size = (int(w), int(h))
    klist = k.flatten().tolist()
    klist = klist[: size[0] * size[1]]
    return img.filter(ImageFilter.Kernel(size=size, kernel=klist, scale=1.0))