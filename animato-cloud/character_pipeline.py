#!/usr/bin/env python3
"""AI character -> animation rig pipeline.

One complete character image (generated with the app's image generator, or imported by
the user) is cut into reusable transparent layers:

  1. subject matte  — the PNG's own transparency, else rembg human matting
  2. body pose      — YOLOv8-pose keypoints (shoulders/elbows/wrists/hips/ankles) decide
                      portrait vs full body and where the arms and hands are
  3. face parsing   — CelebAMask-HQ classes; small faces (full-body shots) are re-parsed
                      on a zoomed head crop so eyes/brows/lips are found reliably
  4. repair         — DreamShaper inpainting for revealed areas (scalp under hair) and for
                      variants (closed eyes, open/round mouth, hand poses); every variant
                      has a CPU fallback so a rig ALWAYS blinks, talks and gestures
  5. rig manifest   — layers ordered front→back for the canvas engine, centred units,
                      neck/shoulder/elbow/wrist pivots, arm FK metadata, 9 visemes.

Every stage is defensive: a missing class (no eyes found, no hands in frame, no neck) only
removes that layer — it never stops the run.
"""
import argparse, json, math, os, sys, time, traceback, urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("YOLO_VERBOSE", "False")


def _patch_pil_shapes():
    """PIL raises ValueError when x1<x0 or y1<y0. Odd crops (waist-up portraits, tiny
    silhouettes) can produce those boxes, so normalise them instead of crashing the rig."""
    from PIL import ImageDraw
    def norm(xy):
        v = list(xy)
        if len(v) == 2 and hasattr(v[0], "__len__"): v = [v[0][0], v[0][1], v[1][0], v[1][1]]
        if len(v) == 4:
            x0, y0, x1, y1 = v
            v = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        return v
    for name in ("rectangle", "ellipse"):
        orig = getattr(ImageDraw.ImageDraw, name)
        def wrap(self, xy, *a, _o=orig, **k): return _o(self, norm(xy), *a, **k)
        setattr(ImageDraw.ImageDraw, name, wrap)
_patch_pil_shapes()

PARTS = {
    "background": 0, "skin": 1, "l_brow": 2, "r_brow": 3, "l_eye": 4, "r_eye": 5, "eyeglass": 6,
    "l_ear": 7, "r_ear": 8, "earring": 9, "nose": 10, "mouth": 11, "u_lip": 12, "l_lip": 13,
    "neck": 14, "neck_l": 15, "cloth": 16, "hair": 17, "hat": 18,
}
FACE_CLASSES = list(range(1, 14))
HEAD_CLASSES = list(range(1, 16)) + [17, 18]
FACE_MODEL_REPO = os.environ.get("CHARACTER_FACE_MODEL", "PayamFard123/dermaintel-face-parsing")
FACE_MODEL_FILE = os.environ.get("CHARACTER_FACE_MODEL_FILE", "resnet18.onnx")
INPAINT_MODEL = os.environ.get("CHARACTER_INPAINT_MODEL", "Lykon/dreamshaper-8-inpainting")
POSE_MODEL = os.environ.get("CHARACTER_POSE_MODEL", "yolov8n-pose.pt")
CACHE = Path(os.environ.get("CHARACTER_CACHE", str(Path.home() / ".cache/animato-character")))
POSE_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
              "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
HAND_POSES = ("relaxed", "open", "point", "fist")


def log(msg): print(f"[character] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Small, None-safe mask helpers. Every helper accepts None so a class the parser
# did not find (closed eyes, no neck, stylised art) can never crash the run.
# ---------------------------------------------------------------------------
def empty_mask(size):
    from PIL import Image
    return Image.new("L", size, 0)


def has(mask):
    return mask is not None and mask.getbbox() is not None


def union(*masks):
    from PIL import ImageChops
    out = None
    for m in masks:
        if m is None: continue
        out = m if out is None else ImageChops.lighter(out, m)
    return out


def subtract(a, b):
    """a minus b. Returns a unchanged when b is None (the crash in the old pipeline:
    `ImageChops.subtract(eye, None)` when an eye had no detectable iris)."""
    from PIL import ImageChops
    if a is None: return None
    if b is None: return a
    return ImageChops.subtract(a, b)


def intersect(a, b):
    from PIL import ImageChops
    if a is None or b is None: return None
    return ImageChops.multiply(a, b)


def bbox(mask, pad=0, size=None):
    if mask is None: return None
    b = mask.getbbox()
    if not b: return None
    x0, y0, x1, y1 = b
    W, H = size or mask.size
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def dilate(mask, px):
    from PIL import ImageFilter
    if mask is None: return None
    px = int(px)
    if px <= 0: return mask
    out = mask
    # MaxFilter must be odd and large kernels are slow — repeat a capped kernel instead.
    while px > 0:
        step = min(px, 15)
        out = out.filter(ImageFilter.MaxFilter(step * 2 + 1))
        px -= step
    return out


def erode(mask, px):
    from PIL import ImageFilter
    if mask is None or px <= 0: return mask
    return mask.filter(ImageFilter.MinFilter(min(31, int(px) * 2 + 1)))


def binarize(mask, thr=127):
    if mask is None: return None
    return mask.point(lambda v: 255 if v > thr else 0)


def shape_mask(size, kind, box):
    from PIL import Image, ImageDraw
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    (d.ellipse if kind == "ellipse" else d.rectangle)([int(v) for v in box], fill=255)
    return m


def capsule(size, a, b, r):
    """Filled stadium from point a to point b with radius r (a limb segment)."""
    from PIL import Image, ImageDraw
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.line([tuple(map(float, a)), tuple(map(float, b))], fill=255, width=max(1, int(r * 2)))
    for p in (a, b):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    return m


def largest_component(mask, near=None):
    """Keep the connected component nearest to `near` (or the largest one)."""
    try:
        import cv2, numpy as np
        from PIL import Image
        a = (np.array(mask) > 127).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(a, 8)
        if n <= 1: return mask
        best, score = 1, None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            s = -area
            if near is not None:
                cx, cy = cent[i]
                s = math.hypot(cx - near[0], cy - near[1]) - math.sqrt(area) * 0.5
            if score is None or s < score: best, score = i, s
        return Image.fromarray(((lab == best) * 255).astype(np.uint8), "L")
    except Exception:
        return mask


def main_components(mask, keep_ratio=0.015):
    """Drop specks/noise from a matte but keep every sizeable piece (a hand separated by
    a thin gap, shoes, a hat brim) — not just the single largest blob."""
    try:
        import cv2, numpy as np
        from PIL import Image
        a = (np.array(mask) > 127).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
        if n <= 2: return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, ar in enumerate(areas) if ar >= areas.max() * keep_ratio]
        return Image.fromarray((np.isin(lab, keep) * 255).astype(np.uint8), "L")
    except Exception:
        return mask


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_pil(path_or_bytes):
    from PIL import Image, ImageOps
    if isinstance(path_or_bytes, (str, Path)): im = Image.open(path_or_bytes)
    else:
        import io
        im = Image.open(io.BytesIO(path_or_bytes))
    try: im = ImageOps.exif_transpose(im)
    except Exception: pass
    return im.convert("RGBA")


def download(url, out, max_bytes=20 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "Animato-Character-Rig/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r, open(out, "wb") as f:
        total = 0
        while True:
            b = r.read(1024 * 1024)
            if not b: break
            total += len(b)
            if total > max_bytes: raise RuntimeError("source image is too large")
            f.write(b)
    return out


def extract(img, mask, out_file, pad=3):
    """Cut `mask` out of `img` into a cropped transparent PNG."""
    from PIL import Image
    b = bbox(mask, pad, img.size)
    if not b: return None
    # Straight alpha: keep the true colours and use the mask as alpha. (Pasting through a
    # mask onto transparent black darkened every soft edge into a grey fringe.)
    from PIL import ImageChops
    layer = img.convert("RGBA").copy()
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask.convert("L")))
    layer = layer.crop(b)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    layer.save(out_file, "PNG", optimize=True, compress_level=6)
    return {"file": str(Path(out_file).as_posix()), "bbox": [b[0], b[1], b[2], b[3]], "width": b[2] - b[0], "height": b[3] - b[1]}


def bone_layer(src, mask, joint, end, out_file, pad=4):
    """Cut a limb segment and rotate it so the bone points straight down (+y) with the
    joint at the top. The FK arm rig (applyArmPose) rotates from that rest direction."""
    import numpy as np, cv2
    from PIL import Image
    if not has(mask): return None
    rgba = np.array(src.convert("RGBA"))
    rgba[..., 3] = np.minimum(rgba[..., 3], np.array(mask))
    dx, dy = end[0] - joint[0], end[1] - joint[1]
    if math.hypot(dx, dy) < 1: dx, dy = 0.0, 1.0
    th = math.pi / 2 - math.atan2(dy, dx)
    c, s = math.cos(th), math.sin(th)
    b = mask.getbbox()
    reach = max(math.hypot(x - joint[0], y - joint[1]) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])))
    S = int(2 * (reach + pad)) + 2
    M = np.array([[c, -s, S / 2 - (c * joint[0] - s * joint[1])],
                  [s, c, S / 2 - (s * joint[0] + c * joint[1])]], np.float32)
    out = cv2.warpAffine(rgba, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ys, xs = np.where(out[..., 3] > 8)
    if not len(xs): return None
    x0, x1 = max(0, min(int(xs.min()) - pad, S // 2 - 2)), min(S, max(int(xs.max()) + pad + 1, S // 2 + 2))
    y0, y1 = max(0, min(int(ys.min()) - pad, S // 2 - 2)), min(S, max(int(ys.max()) + pad + 1, S // 2 + 2))
    crop = out[y0:y1, x0:x1]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop, "RGBA").save(out_file, "PNG", optimize=True, compress_level=6)
    w, h = x1 - x0, y1 - y0
    return {"file": str(Path(out_file).as_posix()), "width": w, "height": h, "jx": S / 2 - x0, "jy": S / 2 - y0,
            "length": math.hypot(end[0] - joint[0], end[1] - joint[1])}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
_REMBG = {}
MATTE_MODELS = [m.strip() for m in os.environ.get("CHARACTER_MATTE_MODELS", "isnet-anime,u2net_human_seg").split(",") if m.strip()]
MATTE_BACKUP = os.environ.get("CHARACTER_MATTE_BACKUP", "isnet-general-use")


def _rembg(img, model):
    from rembg import remove, new_session
    if model not in _REMBG: _REMBG[model] = new_session(model)
    out = remove(img.convert("RGB"), session=_REMBG[model], alpha_matting=False)
    return out.getchannel("A").convert("L")


def matte_score(img, m, pose):
    """How believable a matte is: keypoints inside it, its outline following real image
    edges, not leaking into the frame border, not shattered into pieces."""
    import numpy as np, cv2
    a = np.array(m) > 127
    H, W = a.shape
    frac = a.mean()
    if frac < 0.02 or frac > 0.97: return -9.0
    score = 0.0
    if pose:
        pts = [(x, y) for (x, y, c) in pose.values() if c >= 0.4 and 0 <= x < W and 0 <= y < H]
        if pts:
            grown = cv2.dilate(a.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            score += 3.0 * (sum(1 for x, y in pts if grown[int(y), int(x)]) / len(pts))
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    u8 = a.astype(np.uint8)
    edge = (cv2.dilate(u8, np.ones((3, 3), np.uint8)) - cv2.erode(u8, np.ones((3, 3), np.uint8))) > 0
    if edge.any(): score += min(2.5, float(mag[edge].mean()) / (float(mag.mean()) + 1e-3)) * 0.8
    border = np.concatenate([a[0, :], a[:, 0], a[:, -1]])   # bottom edge excluded: crops touch it
    score -= 1.2 * float(border.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    big = (stats[1:, cv2.CC_STAT_AREA] > a.size * 0.004).sum() if n > 1 else 0
    score -= 0.15 * max(0, int(big) - 1)
    return score


def subject_mask(img, pose=None, models=None):
    """Soft character matte. Several matting models are tried (an anime/illustration model,
    a human-photo model and a general one) and the most believable matte wins, so photos,
    3D renders and drawn characters are all cut cleanly. None when rembg is unavailable."""
    cands = []
    for model in (models or MATTE_MODELS):
        try:
            t = time.time(); m = _rembg(img, model)
            if m.getbbox(): cands.append((model, m))
            log(f"matte {model}: {time.time() - t:.1f}s")
        except Exception as e:
            log(f"matte {model} unavailable ({str(e)[:100]})")
    if not cands: return None
    if len(cands) == 1 and models: return cands[0][1]
    try:
        scored = sorted(((matte_score(img, m, pose), name, m) for name, m in cands), key=lambda x: -x[0])
        # Both default mattes look poor → also try the general model (costs a few seconds).
        if models is None and MATTE_BACKUP and scored[0][0] < 2.0:
            try:
                bm = _rembg(img, MATTE_BACKUP)
                if bm.getbbox(): scored = sorted(scored + [(matte_score(img, bm, pose), MATTE_BACKUP, bm)], key=lambda x: -x[0])
            except Exception as e: log(f"matte {MATTE_BACKUP} unavailable ({str(e)[:80]})")
        log("matte scores: " + ", ".join(f"{n}={sc:.2f}" for sc, n, _ in scored))
        return scored[0][2]
    except Exception as e:
        log(f"matte scoring failed ({str(e)[:100]}) — using {cands[0][0]}")
        return cands[0][1]


class FaceParser:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        import shutil
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / FACE_MODEL_FILE
        # A previous run may have left a dangling symlink (HF cache links are relative).
        if p.is_symlink() and not p.exists(): p.unlink()
        if p.exists() and p.stat().st_size == 0: p.unlink()
        if not p.exists():
            log("downloading the face parser weights…")
            src = hf_hub_download(repo_id=FACE_MODEL_REPO, filename=FACE_MODEL_FILE, cache_dir=str(CACHE))
            real = Path(src).resolve()
            if not real.exists() or real.stat().st_size == 0:
                raise FileNotFoundError(f"face parser weights missing after download: {src}")
            shutil.copy2(real, p)
        log(f"face parser weights: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def parse(self, img):
        import numpy as np
        from PIL import Image
        rgb = img.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
        x = ((arr - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        y = self.session.run(None, {self.input: x})[0]
        if y.ndim == 4: y = y[0]
        labels = np.argmax(y, axis=0).astype(np.uint8)
        return Image.fromarray(labels, "L").resize(img.size, Image.Resampling.NEAREST)


# 68 face landmarks (iBUG order) of the main person, filled by detect_pose when the
# whole-body model is available: {"face": [(x, y, score) * 68]}.
LANDMARKS = {}


def _pose_from_arrays(kps, scores):
    """Pick the biggest person from (N,K,2) keypoints + (N,K) scores → (pose17, index)."""
    import numpy as np
    kps, scores = np.asarray(kps, np.float32), np.asarray(scores, np.float32)
    if kps.ndim != 3 or kps.shape[0] == 0 or kps.shape[1] < 17: return None
    best, area = 0, -1.0
    for i in range(kps.shape[0]):
        good = scores[i, :17] > 0.3
        if good.sum() < 3: continue
        pts = kps[i, :17][good]
        a = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
        if a > area: best, area = i, a
    if area < 0: return None
    return {POSE_NAMES[j]: (float(kps[best, j, 0]), float(kps[best, j, 1]), float(min(1.0, scores[best, j]))) for j in range(17)}, best


def _face_points(kps, scores, idx):
    """Face landmarks (COCO-WholeBody points 23..90) of person `idx`, or None."""
    import numpy as np
    k, s = np.asarray(kps, np.float32), np.asarray(scores, np.float32)
    if k.ndim != 3 or k.shape[1] < 91: return None
    return [(float(x), float(y), float(c)) for (x, y), c in zip(k[idx, 23:91], s[idx, 23:91])]


def _face_quality(face):
    import numpy as np
    if not face: return 0.0
    sc = np.array([c for _, _, c in face], np.float32)
    # eyes (36-47) and mouth (48-67) matter most for the rig
    return float(np.median(np.concatenate([sc[36:48], sc[48:68]])))


def detect_pose(img):
    """17 COCO keypoints of the main person: {name: (x, y, conf)} or None.
    RTMW whole-body via rtmlib (onnxruntime, no PyTorch): the same body keypoints plus
    68 face landmarks (stored in LANDMARKS["face"]) that locate the eyes, brows and lips
    on illustrated, 3D and anime faces where a photo-trained face parser finds nothing.
    Falls back to the body-only model, then YOLOv8-pose."""
    LANDMARKS.clear()
    if os.environ.get("CHARACTER_ENABLE_POSE", "1") == "0": return None
    import numpy as np
    t = time.time()
    bgr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    try:
        import rtmlib
        model, kind = None, "body"
        if os.environ.get("CHARACTER_POSE_KIND", "wholebody") == "wholebody":
            try:
                model = rtmlib.Wholebody(mode=os.environ.get("CHARACTER_WHOLEBODY_MODE", "balanced"), to_openpose=False, backend="onnxruntime", device="cpu")
                kind = "wholebody"
            except Exception as e:
                log(f"whole-body model unavailable ({str(e)[:120]}) — body keypoints only")
        if model is None:
            model = rtmlib.Body(mode=os.environ.get("CHARACTER_POSE_MODE", "balanced"), to_openpose=False, backend="onnxruntime", device="cpu")
        kps, scores = model(bgr)
        res = _pose_from_arrays(kps, scores)
        if not res:
            log("pose: no person found")
            return None
        pts, idx = res
        if kind == "wholebody":
            face = _face_points(kps, scores, idx)
            try:
                # Small face (full-body shot): run the landmark model again on a head-and-
                # shoulders crop so the face fills more of its input.
                if face:
                    fx = [p[0] for p in face]; fy = [p[1] for p in face]
                    fwid = max(fx) - min(fx)
                    if fwid > 4 and fwid < min(img.size) * 0.3:
                        cx, cy = (max(fx) + min(fx)) / 2, (max(fy) + min(fy)) / 2
                        half = fwid * 1.6
                        box = [max(0, cx - half), max(0, cy - half * 1.2), min(img.width, cx + half), min(img.height, cy + half * 1.5)]
                        k2, s2 = model.pose_model(bgr, bboxes=[box])
                        face2 = _face_points(k2, s2, 0)
                        if face2 and _face_quality(face2) >= _face_quality(face) * 0.9: face = face2
            except Exception as e:
                log(f"zoomed landmark pass skipped ({str(e)[:100]})")
            if face:
                LANDMARKS["face"] = face
                log(f"face landmarks: quality {_face_quality(face):.2f}")
        log(f"pose ({kind}, {time.time() - t:.1f}s): " + ", ".join(f"{k}={v[2]:.2f}" for k, v in pts.items()))
        return pts
    except ImportError:
        pass
    except Exception as e:
        log(f"rtmpose failed ({str(e)[:140]}) — trying YOLOv8-pose")
    try:
        from ultralytics import YOLO
        cached = CACHE / POSE_MODEL
        model = YOLO(str(cached) if cached.exists() and cached.stat().st_size > 0 else POSE_MODEL)
        res = model.predict(img.convert("RGB"), verbose=False, device="cpu", imgsz=640, conf=0.2)
        if not res or res[0].keypoints is None: return None
        kp = res[0].keypoints.data.cpu().numpy()
        r = _pose_from_arrays(kp[..., :2], kp[..., 2] if kp.shape[-1] > 2 else np.ones(kp.shape[:2]))
        if r: log(f"pose (yolov8, {time.time() - t:.1f}s)")
        return r[0] if r else None
    except Exception as e:
        log(f"pose model unavailable ({str(e)[:120]}) — using silhouette geometry")
        return None


def inpaint_pipe():
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(INPAINT_MODEL, torch_dtype=torch.float32, safety_checker=None, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True); pipe.to("cpu")
    for fn in ("enable_attention_slicing", "enable_vae_slicing"):
        try: getattr(pipe, fn)()
        except Exception: pass
    return pipe


INPAINT_T0 = time.time()
def inpaint_budget_left():
    """CPU diffusion is slow; stop starting new inpaints once the time budget is used up."""
    return float(os.environ.get("CHARACTER_INPAINT_BUDGET", "900")) - (time.time() - INPAINT_T0)


def rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"): return int(line.split()[1]) // 1024
    except Exception: pass
    return -1


def inpaint_region(pipe, image, mask, prompt, seed=1, steps=None):
    """Localized inpaint that only replaces pixels inside the repair mask."""
    from PIL import Image, ImageFilter
    import torch
    b = bbox(mask, 18, image.size)
    if not b: return image
    original_crop = image.crop(b).convert("RGB")
    original_size = original_crop.size
    original_mask = mask.crop(b).convert("L").filter(ImageFilter.GaussianBlur(1.2))
    crop, m = original_crop, original_mask
    max_side = int(os.environ.get("CHARACTER_INPAINT_MAX_SIDE", "384"))
    scale = max_side / max(crop.size)
    size = (max(256, int(crop.width * scale) // 8 * 8), max(256, int(crop.height * scale) // 8 * 8))
    crop = crop.resize(size, Image.Resampling.LANCZOS); m = m.resize(size, Image.Resampling.BILINEAR)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 32))
    steps = int(steps or os.environ.get("CHARACTER_INPAINT_STEPS", "20"))
    log(f"inpaint start: {crop.size[0]}x{crop.size[1]}, {steps} steps, {rss_mb()} MB RAM"); _t = time.time()
    with torch.inference_mode():
        out = pipe(prompt=prompt, negative_prompt="deformed, extra fingers, blurry, distorted, cartoon outline, text",
                   image=crop, mask_image=m, num_inference_steps=steps,
                   guidance_scale=float(os.environ.get("CHARACTER_INPAINT_GUIDANCE", "7")), generator=gen).images[0].convert("RGB")
    log(f"inpaint done in {time.time() - _t:.1f}s, {rss_mb()} MB RAM")
    if out.size != original_size: out = out.resize(original_size, Image.Resampling.LANCZOS)
    blended = original_crop.convert("RGBA")
    blended.paste(out.convert("RGBA"), (0, 0), original_mask)
    base = image.copy(); base.paste(blended, (b[0], b[1])); return base


def cv_inpaint(image, mask, radius=5):
    """Fast CPU fill (Telea) — used for hidden skin under features and behind arms."""
    if not has(mask): return image
    try:
        import cv2, numpy as np
        from PIL import Image
        rgb = np.array(image.convert("RGB")); mm = (np.array(mask) > 0).astype(np.uint8) * 255
        fixed = cv2.inpaint(rgb, mm, radius, cv2.INPAINT_TELEA)
        out = Image.fromarray(fixed).convert("RGBA"); out.putalpha(image.getchannel("A")); return out
    except Exception:
        return image


# ---------------------------------------------------------------------------
# Face parsing with a zoomed head crop
# ---------------------------------------------------------------------------
def ok(pose, k, thr=0.35, size=None):
    if not pose or k not in pose: return False
    x, y, c = pose[k]
    if c < thr: return False
    if size and not (0 <= x < size[0] and 0 <= y < size[1]): return False
    return True


def parse_labels(parser, img, pose):
    import numpy as np
    from PIL import Image
    W, H = img.size
    labels = np.array(parser.parse(img))
    face = np.isin(labels, FACE_CLASSES)
    box = None
    ys, xs = np.where(face)
    if len(xs) > 60:
        box = [float(np.percentile(xs, 1)), float(np.percentile(ys, 1)), float(np.percentile(xs, 99)), float(np.percentile(ys, 99))]
    pbox = None
    if pose:
        pts = [pose[k] for k in ("nose", "l_eye", "r_eye", "l_ear", "r_ear") if ok(pose, k, 0.4, (W, H))]
        if len(pts) >= 2:
            px = [p[0] for p in pts]; py = [p[1] for p in pts]
            sd = math.hypot(pose["l_shoulder"][0] - pose["r_shoulder"][0], pose["l_shoulder"][1] - pose["r_shoulder"][1]) \
                if ok(pose, "l_shoulder") and ok(pose, "r_shoulder") else 0
            size = max((max(px) - min(px)) * 1.9, sd * 0.5, 24)
            cx, cy = sum(px) / len(px), sum(py) / len(py)
            pbox = [cx - size / 2, cy - size * 0.6, cx + size / 2, cy + size * 0.65]
    if pbox is not None:
        def area(b): return max(1, (b[2] - b[0]) * (b[3] - b[1]))
        inter = max(0, min(box[2], pbox[2]) - max(box[0], pbox[0])) * max(0, min(box[3], pbox[3]) - max(box[1], pbox[1])) if box else 0
        if box is None or inter < 0.25 * area(pbox) or area(box) > 5 * area(pbox): box = pbox
    zoomed = False
    if box is not None:
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 0.42 * W or bh < 0.3 * H:
            side = max(bw, bh) * 2.1
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + side * 0.08
            c = [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(W, cx + side / 2)), int(min(H, cy + side / 2))]
            if c[2] - c[0] > 24 and c[3] - c[1] > 24:
                sub = np.array(parser.parse(img.crop(c)))
                coarse_hair = labels == PARTS["hair"]
                labels = np.where(np.isin(labels, HEAD_CLASSES), 0, labels)
                # Long hair can continue below the head crop — keep the coarse hair near it.
                keep = np.zeros_like(coarse_hair)
                sw = c[2] - c[0]
                keep[c[1]:min(H, int(c[3] + sw * 0.9)), max(0, int(c[0] - sw * 0.25)):min(W, int(c[2] + sw * 0.25))] = True
                labels[coarse_hair & keep] = PARTS["hair"]
                region = labels[c[1]:c[3], c[0]:c[2]]
                labels[c[1]:c[3], c[0]:c[2]] = np.where(sub > 0, sub, region)
                zoomed = True
                log(f"face is small ({int(bw)}x{int(bh)}px) — re-parsed a zoomed head crop {c}")
    return Image.fromarray(labels.astype(np.uint8), "L"), zoomed


def mask_for(labels, *names):
    from PIL import Image
    import numpy as np
    a = np.asarray(labels, dtype=np.uint8)
    return Image.fromarray(np.isin(a, [PARTS[n] for n in names]).astype(np.uint8) * 255, "L")


# ---------------------------------------------------------------------------
# Eyes, blinks and mouths
# ---------------------------------------------------------------------------
def split_eye_details(eye_mask, image):
    """eye mask -> (eye, iris, pupil, iris_src).

    Works for photos and for illustrated / anime eyes (large bright irises): the iris is
    the part of the eye whose colour differs from the eye white, completed into a full
    circle (the lids hide part of it in the picture); the pupil is the darkest blob in it.
    `iris_src` is the image with the hidden part of the iris painted in, so the iris can
    move under the eyelid frame without showing a flat, clipped edge."""
    import numpy as np
    from PIL import Image
    try: import cv2
    except Exception: cv2 = None
    if not has(eye_mask): return None, None, None, None
    x0, y0, x1, y1 = eye_mask.getbbox()
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    if w < 6 or h < 4 or cv2 is None: return eye_mask, None, None, None
    pad = int(max(w, h) * 0.6)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)
    rgb = np.array(image.convert("RGB").crop((X0, Y0, X1, Y1)))
    eye = (np.array(eye_mask.crop((X0, Y0, X1, Y1))) > 127).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    inner = cv2.erode(eye, np.ones((3, 3), np.uint8), iterations=max(1, int(h * 0.1)))
    if inner.sum() < 6: inner = eye
    px = lab[inner > 0]
    sclera = np.median(px[px[:, 0] >= np.percentile(px[:, 0], 65)], axis=0)
    dist = np.sqrt(((lab - sclera) ** 2).sum(axis=2))
    cand = ((dist > 26) | (L < sclera[0] * 0.72)).astype(np.uint8) & inner
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labm, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    ecx, ecy = (x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0
    best, bscore = 0, None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max(4, w * h * 0.02): continue
        sc = -area + (abs(cent[i][0] - ecx) / w + abs(cent[i][1] - ecy) / h) * w * h * 0.3
        if bscore is None or sc < bscore: best, bscore = i, sc
    if best:
        bx, by, bw, bh = stats[best, 0], stats[best, 1], stats[best, 2], stats[best, 3]
        r = float(np.clip(max(bw, bh * 1.05) / 2, h * 0.32, w * 0.4))
        cx = bx + bw / 2
        # The lids clip the iris: its visible bottom edge locates the centre.
        cy = (by + bh - r) if bh < 2 * r * 0.95 and (by + bh) >= (y1 - Y0) - h * 0.25 else by + bh / 2
        cy = float(np.clip(cy, ecy - h * 0.45, ecy + h * 0.45))
    else:
        r, cx, cy = float(min(h * 0.5, w * 0.24)), ecx, ecy
    iris = np.zeros(eye.shape, np.uint8); cv2.circle(iris, (int(round(cx)), int(round(cy))), max(2, int(round(r))), 1, -1)
    vis = iris & eye
    pupil = np.zeros_like(iris)
    if vis.sum() > 8:
        thr = np.percentile(L[vis > 0], 22)
        dk = ((L <= thr).astype(np.uint8) & vis)
        n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(dk, 8)
        if n2 > 1:
            j = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            pcx, pcy = ce2[j]
            pr = float(np.clip(np.sqrt(st2[j, cv2.CC_STAT_AREA] / np.pi) * 1.15, r * 0.25, r * 0.55))
            cv2.circle(pupil, (int(round(pcx)), int(round(pcy))), max(1, int(round(pr))), 1, -1)
    if pupil.sum() == 0: cv2.circle(pupil, (int(round(cx)), int(round(cy))), max(1, int(round(r * 0.42))), 1, -1)
    pupil &= iris
    src = rgb.copy()
    hidden = (iris > 0) & (eye == 0)
    if hidden.any() and vis.sum() > 4:
        fill = cv2.inpaint(rgb, (hidden * 255).astype(np.uint8), max(2, int(r * 0.4)), cv2.INPAINT_TELEA)
        med = np.median(rgb[vis > 0], axis=0)
        src[hidden] = (fill[hidden] * 0.5 + med * 0.5).astype(np.uint8)
    def full(local):
        mm = Image.new("L", eye_mask.size, 0); mm.paste(Image.fromarray((local > 0).astype(np.uint8) * 255, "L"), (X0, Y0)); return mm
    srcimg = image.convert("RGBA").copy()
    srcimg.paste(Image.fromarray(src).convert("RGBA"), (X0, Y0))
    iris_m, pupil_m = full(iris), full(pupil)
    return eye_mask, (iris_m if has(iris_m) else None), (pupil_m if has(pupil_m) else None), srcimg


def sclera_fill(img, eye, iris):
    """Eye-white layer source: the iris area is repainted with the sclera colour so the
    moving iris never reveals a second, static iris underneath."""
    import numpy as np
    from PIL import Image, ImageFilter
    if not has(eye) or not has(iris): return img
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    e = np.array(eye) > 127; i = np.array(dilate(iris, 1)) > 127
    white = e & ~i
    px = rgb[white]
    if len(px) < 4: col = np.array([235, 232, 228], np.float32)
    else:
        lum = px.mean(axis=1)
        col = np.median(px[lum >= np.percentile(lum, 60)], axis=0)
    out = rgb.copy(); out[i & e] = col
    res = Image.fromarray(out.clip(0, 255).astype(np.uint8)).convert("RGBA")
    soft = res.filter(ImageFilter.GaussianBlur(0.8))
    res.paste(soft, (0, 0), dilate(intersect(iris, eye), 1))
    return res


def synth_blink(clean, img, eye_mask):
    """Closed-eye fallback: skin from the feature-free face + a curved lash line."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter, Image
    if not has(eye_mask): return None
    x0, y0, x1, y1 = eye_mask.getbbox()
    w, h = x1 - x0, max(2, y1 - y0)
    out = clean.copy()
    reg = np.array(img.convert("RGB").crop((x0, y0, x1, y1))).reshape(-1, 3).astype(np.float32)
    lum = reg.mean(axis=1)
    # Lash colour: the darkest pixels, desaturated and pulled towards a deep brown so an iris
    # colour (green/blue) never tints the closed lid.
    dark = np.median(reg[lum <= np.percentile(lum, 6)], axis=0) if len(reg) else np.array([40, 28, 26], np.float32)
    g = float(dark.mean())
    lash = tuple(int(min(70, v)) for v in (np.array([g, g, g]) * 0.5 + np.array([34, 24, 22]) * 0.5))
    over = Image.new("RGBA", out.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    ym = y0 + h * 0.62; sag = h * 0.22
    pts = [(x0 + w * t, ym + sag * (1 - (2 * t - 1) ** 2)) for t in [k / 16 for k in range(17)]]
    d.line(pts, fill=lash + (240,), width=max(2, int(round(h * 0.16))), joint="curve")
    crease = [(x0 + w * (0.12 + 0.76 * t), y0 + h * 0.18 + sag * 0.6 * (1 - (2 * t - 1) ** 2)) for t in [k / 12 for k in range(13)]]
    d.line(crease, fill=lash + (55,), width=max(1, int(h * 0.08)), joint="curve")
    over = over.filter(ImageFilter.GaussianBlur(max(0.5, h * 0.03)))
    out.alpha_composite(over)
    return out


def synth_mouth(img, lips, inner, kind):
    """Speaking-mouth fallback drawn into the lips when diffusion variants are unavailable."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter, Image
    region = union(lips, inner)
    if not has(region): return None
    x0, y0, x1, y1 = region.getbbox()
    lw, lh = x1 - x0, max(3, y1 - y0)
    cx = (x0 + x1) / 2
    # Seam between the lips: bottom of the upper lip, else the middle of the mouth.
    seam = y0 + lh * 0.5
    if has(inner):
        ib = inner.getbbox(); seam = (ib[1] + ib[3]) / 2
    shapes = {"open": (0.50, 0.62, 0.30), "round": (0.30, 0.72, 0.0), "wide": (0.62, 0.34, 0.45)}
    fw, fh, teeth = shapes.get(kind, shapes["open"])
    ew, eh = lw * fw, max(2.0, lh * fh)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    box = [cx - ew / 2, seam - eh * 0.42, cx + ew / 2, seam + eh * 0.58]
    d.ellipse(box, fill=(52, 16, 20, 245))
    if teeth > 0:
        tb = Image.new("L", img.size, 0); ImageDraw.Draw(tb).rectangle([box[0], box[1], box[2], box[1] + eh * teeth], fill=255)
        el = Image.new("L", img.size, 0); ImageDraw.Draw(el).ellipse([box[0] + ew * 0.06, box[1], box[2] - ew * 0.06, box[3]], fill=255)
        tm = intersect(tb, el)
        teeth_layer = Image.new("RGBA", img.size, (238, 234, 226, 235))
        over.paste(teeth_layer, (0, 0), tm)
    tongue = [cx - ew * 0.28, box[3] - eh * 0.38, cx + ew * 0.28, box[3] - eh * 0.04]
    d.ellipse(tongue, fill=(150, 62, 66, 150))
    over = over.filter(ImageFilter.GaussianBlur(max(0.5, lh * 0.04)))
    out = img.copy(); out.alpha_composite(over)
    return out


# ---------------------------------------------------------------------------
# Rig assembly helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Visemes from the character's OWN lips. Every mouth shape is the original upper and
# lower lip, moved / stretched / curved like a real mouth, with the inside of the mouth,
# teeth and tongue coloured from the character itself. Nothing is pasted from elsewhere,
# so the lip colour, gloss, outline and art style always match the face.
# ---------------------------------------------------------------------------
VISEME_SHAPES = {
    #        gap   inner-w  lip-sx  lip-sy  teeth           tongue
    "REST": (0.00, 0.00,    1.00,   1.00,   "",             0.0),
    "MBP":  (0.00, 0.00,    1.03,   0.90,   "",             0.0),
    "AI":   (0.58, 0.64,    1.00,   1.00,   "upper",        0.35),
    "E":    (0.28, 0.76,    1.10,   0.96,   "both",         0.0),
    "O":    (0.62, 0.46,    0.80,   1.08,   "",             0.25),
    "U":    (0.32, 0.32,    0.70,   1.10,   "",             0.0),
    "L":    (0.48, 0.58,    1.00,   1.00,   "upper",        0.65),
    "CONS": (0.18, 0.66,    1.05,   0.98,   "both",         0.0),
    "FV":   (0.10, 0.60,    1.02,   1.00,   "bite",         0.0),
}
MOODS = {"neutral": (0.0, 1.0), "smile": (0.0, 1.0), "happy": (0.0, 1.0), "sad": (0.0, 1.0)}


def _premul(a):
    import numpy as np
    out = a.astype(np.float32).copy(); al = out[..., 3:4] / 255.0; out[..., :3] *= al; return out


def _unpremul(a):
    import numpy as np
    out = a.copy(); al = np.maximum(out[..., 3:4], 1e-3) / 255.0
    out[..., :3] = np.where(out[..., 3:4] > 0.5, out[..., :3] / al, 0); return out.clip(0, 255).astype(np.uint8)


def _warp_lip(rgba, cx, cy, sx, sy, dy, curve, half_w, lh):
    """Scale a lip about (cx, cy), move it by dy and bend its corners by `curve` (smile > 0)."""
    import numpy as np, cv2
    H, W = rgba.shape[:2]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    t = np.clip((xs - cx) / max(1.0, half_w * sx), -1.2, 1.2)
    bend = -curve * lh * (t ** 2)                         # corners up for a smile
    map_x = cx + (xs - cx) / sx
    map_y = cy + (ys - cy - dy - bend) / sy
    out = cv2.remap(_premul(rgba), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return _unpremul(out)


def _over(base, top):
    """Alpha-composite straight-alpha RGBA uint8 arrays."""
    import numpy as np
    a = top[..., 3:4].astype(np.float32) / 255.0
    out = base.astype(np.float32).copy()
    out[..., :3] = top[..., :3] * a + out[..., :3] * (1 - a)
    out[..., 3:4] = np.maximum(out[..., 3:4], top[..., 3:4])
    return out.clip(0, 255).astype(np.uint8)


def build_visemes(img, face_clean, u_lip, l_lip, inner, region, teeth_hint=None):
    """Return {mood: {shape: PIL RGBA full-size image}} made from the character's own lips.
    `region` is the mouth layer mask; everything outside it is untouched."""
    import numpy as np, cv2
    from PIL import Image, ImageFilter
    lips = union(u_lip, l_lip)
    if not has(lips) or not has(region): return {}
    W, H = img.size
    rb = bbox(region, 4, img.size)
    x0, y0, x1, y1 = rb
    src = np.array(img.convert("RGBA"))[y0:y1, x0:x1]
    clean = np.array(face_clean.convert("RGBA"))[y0:y1, x0:x1]
    clean[..., 3] = 255
    lb = lips.getbbox(); lw, lh = lb[2] - lb[0], max(3, lb[3] - lb[1])
    cx = (lb[0] + lb[2]) / 2 - x0
    # Split into upper / lower lip. Missing classes → split the lips at the seam.
    if has(inner): ib = inner.getbbox(); seam = (ib[1] + ib[3]) / 2
    elif has(u_lip) and has(l_lip): seam = (u_lip.getbbox()[3] + l_lip.getbbox()[1]) / 2
    else: seam = lb[1] + lh * 0.48
    if not (has(u_lip) and has(l_lip)):
        top = shape_mask(img.size, "rect", [0, 0, W, seam]); u_lip = intersect(lips, top); l_lip = subtract(lips, top)
    seam_c = seam - y0
    # A mouth that is already open in the picture (a grin with teeth): shapes are measured
    # from that opening, so "closed" shapes really close it and open ones open it further.
    open0 = float(inner.getbbox()[3] - inner.getbbox()[1]) if has(inner) else 0.0
    lt = max(3.0, lh - open0)
    src_open = open0 > lt * 0.2
    def lip_rgba(mask):
        m = dilate(mask, 1).filter(ImageFilter.GaussianBlur(0.7))
        a = np.array(m)[y0:y1, x0:x1]
        out = src.copy(); out[..., 3] = np.minimum(out[..., 3], a); return out
    up_rgba, lo_rgba = lip_rgba(union(u_lip, intersect(inner, shape_mask(img.size, "rect", [0, 0, W, seam])) if has(inner) and not src_open else u_lip)), lip_rgba(l_lip)
    both_rgba = lip_rgba(union(u_lip, l_lip))
    inner_w = float(inner.getbbox()[2] - inner.getbbox()[0]) if has(inner) else lw * 0.6
    # Colours from the character itself.
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    lipc = np.median(rgb[np.array(lips) > 127], axis=0)
    innerc = None
    if has(inner):
        px = rgb[np.array(inner) > 127]
        if len(px) >= 15:
            lum = px.mean(1); innerc = np.median(px[lum <= np.percentile(lum, 50)], axis=0)
            bright = px[lum >= 150]
            if len(bright) >= 6 and teeth_hint is None: teeth_hint = np.median(bright, axis=0)
    if innerc is None: innerc = lipc * 0.28 + np.array([18, 4, 8])
    innerc = np.minimum(innerc, lipc * 0.45 + 10)                     # always reads as an opening
    teethc = np.array(teeth_hint if teeth_hint is not None else (236, 232, 224), np.float32) * 0.97
    tonguec = lipc * 0.55 + np.array([190, 80, 92]) * 0.45
    reg = np.array(region)[y0:y1, x0:x1]
    hh, ww = reg.shape
    results = {}
    for mood, (curve, moodsx) in list(MOODS.items())[:1]:
        shapes = {}
        for shape, (gap, iw, lsx, lsy, teeth, tongue) in VISEME_SHAPES.items():
            if shape == "REST" and mood == "neutral":
                base = src.copy()                                     # pixel-exact original
            else:
                g = gap * lt
                sx = lsx * moodsx
                delta = g - open0
                up_dy, lo_dy = -0.22 * delta, 0.78 * delta
                if shape == "MBP": up_dy, lo_dy = 0.22 * open0 + 0.05 * lt, -0.78 * open0 - 0.06 * lt
                if shape == "FV": lo_dy = -0.78 * open0 - 0.06 * lt
                base = clean.copy()
                if g > 0.5 or shape == "FV":
                    # Opening between the lips (drawn first; lips overlap its edges).
                    canvas = np.zeros((hh, ww, 4), np.uint8)
                    ew, eh = max(2.0, lw * iw * moodsx), max(2.0, g + lt * 0.30)
                    ecy = seam_c if src_open else seam_c + (lo_dy + up_dy) / 2 - curve * lh * 0.15
                    ell = np.zeros((hh, ww), np.uint8)
                    cv2.ellipse(ell, (int(cx), int(ecy)), (int(ew / 2), int(eh / 2)), 0, 0, 360, 255, -1)
                    canvas[ell > 0, :3] = innerc; canvas[ell > 0, 3] = 245
                    if tongue > 0:
                        tm = np.zeros_like(ell)
                        cv2.ellipse(tm, (int(cx), int(ecy + eh * (0.5 - tongue * 0.45))), (int(ew * 0.32), int(eh * 0.3)), 0, 0, 360, 255, -1)
                        tm = np.minimum(tm, ell); canvas[tm > 0, :3] = tonguec
                    if teeth in ("upper", "both"):
                        tb = np.zeros_like(ell); tb[: int(ecy - eh / 2 + eh * 0.30)] = 255
                        tb = np.minimum(tb, ell); canvas[tb > 0, :3] = teethc
                    if teeth == "both":
                        tb = np.zeros_like(ell); tb[int(ecy + eh / 2 - eh * 0.22):] = 255
                        tb = np.minimum(tb, ell); canvas[tb > 0, :3] = teethc * 0.95
                    if shape == "FV":
                        canvas = np.zeros((hh, ww, 4), np.uint8)
                    canvas = cv2.GaussianBlur(canvas, (0, 0), max(0.6, lh * 0.03))
                    base = _over(base, canvas)
                if src_open:
                    # A grin: the lips form a ring round the teeth. Sliding the two halves
                    # together crosses their curves, so the whole ring is flattened towards the
                    # seam instead (keeps the smile's curve, closes or opens the gap).
                    k = (lt + g) / (lt + open0)
                    ring = _warp_lip(both_rgba, cx, seam_c, sx, max(0.15, k) * lsy, 0, 0, lw / 2, lh)
                    if shape in ("MBP", "FV") or g <= 0.5:
                        line = np.zeros((hh, ww, 4), np.uint8)
                        cv2.ellipse(line, (int(cx), int(seam_c)), (int(max(2, inner_w * sx * 0.45)), max(1, int(open0 * k * 0.5))), 0, 0, 360,
                                    (*[int(v) for v in innerc], 235), -1)
                        base = _over(base, cv2.GaussianBlur(line, (0, 0), max(0.6, lt * 0.04)))
                    base = _over(base, ring)
                    if shape == "FV":
                        tm = np.zeros((hh, ww, 4), np.uint8)
                        cv2.ellipse(tm, (int(cx), int(seam_c)), (int(lw * 0.22), int(max(2, lt * 0.12))), 0, 0, 180, (*[int(v) for v in teethc], 235), -1)
                        base = _over(base, cv2.GaussianBlur(tm, (0, 0), 0.6))
                    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    full.paste(Image.fromarray(base, "RGBA"), (x0, y0))
                    shapes[shape] = full
                    continue
                lower = _warp_lip(lo_rgba, cx, seam_c, sx, lsy, lo_dy, curve * 0.6, lw / 2, lh)
                base = _over(base, lower)
                if shape == "FV":
                    # Upper teeth resting on the lower lip.
                    tm = np.zeros((hh, ww, 4), np.uint8)
                    cv2.ellipse(tm, (int(cx), int(seam_c + lh * 0.05)), (int(lw * 0.24), int(lh * 0.14)), 0, 0, 180, (*[int(v) for v in teethc], 240), -1)
                    base = _over(base, cv2.GaussianBlur(tm, (0, 0), 0.6))
                upper = _warp_lip(up_rgba, cx, seam_c, sx, lsy, up_dy, curve, lw / 2, lh)
                base = _over(base, upper)
            full = Image.new("RGBA", img.size, (0, 0, 0, 0))
            full.paste(Image.fromarray(base, "RGBA"), (x0, y0))
            shapes[shape] = full
        results[mood] = shapes
    return results


# ---------------------------------------------------------------------------
# Lifelike variants (mouth shapes, closed eyes, hand poses, scalp) with the app's OWN
# image model — the detailed DreamShaper-8 LCM used by imagegen.py — in fast 4-step
# mode. Each variant is repainted IN PLACE on the original pixels inside a tight mask
# (so nothing moves or needs aligning), then Poisson-blended into the real skin so the
# new mouth has the same lighting and colour as the face around it. No warping.
# ---------------------------------------------------------------------------
EDIT_MODEL = os.environ.get("CHARACTER_EDIT_MODEL", "Lykon/dreamshaper-8-lcm")
EDIT_SIZE = int(os.environ.get("CHARACTER_EDIT_SIZE", "320")) // 8 * 8
EDIT_BATCH = int(os.environ.get("CHARACTER_EDIT_BATCH", "4"))
VISEME_PROMPTS = {
    # shape: (what the mouth does, denoise strength)
    "AI": ("mouth open, saying 'ah', jaw relaxed and slightly dropped, upper teeth and the dark inside of the mouth visible", 0.92),
    "E": ("lips stretched wide saying 'ee', mouth barely open, upper and lower teeth slightly visible", 0.85),
    "O": ("lips rounded into an 'O' shape saying 'oh', small dark round opening", 0.92),
    "U": ("lips tightly rounded and pushed forward saying 'oo', very small opening", 0.88),
    "L": ("mouth slightly open, tip of the tongue touching the upper teeth, saying 'l'", 0.88),
    "CONS": ("teeth close together, lips slightly apart, saying 's'", 0.8),
    "FV": ("upper teeth resting on the lower lip, saying 'f'", 0.85),
}
# When a shape is rejected, the closest good shape stands in (never a warped copy).
VISEME_FALLBACK = {"MBP": ["REST"], "AI": ["L", "O", "E"], "E": ["CONS", "AI"], "O": ["U", "AI"], "U": ["O", "AI"],
                   "L": ["AI", "CONS"], "CONS": ["E", "FV", "AI"], "FV": ["CONS", "E", "AI"]}
HAND_PROMPTS = {
    "open": "open hand, palm facing the viewer, relaxed slightly spread fingers",
    "point": "hand pointing with the index finger extended, other fingers curled",
    "fist": "hand in a loose relaxed closed fist",
}
EDIT_NEGATIVE = "deformed, distorted mouth, extra teeth, blurry, smudged, cartoon outline, text, watermark, ugly, low quality"


def load_edit_pipe():
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    t = time.time()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(EDIT_MODEL, torch_dtype=torch.float32, safety_checker=None,
                                                          requires_safety_checker=False, low_cpu_mem_usage=True)
    if "lcm" in EDIT_MODEL.lower():
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe.to("cpu")
    log(f"image model {EDIT_MODEL} ready in {time.time() - t:.1f}s ({rss_mb()} MB RAM)")
    return pipe


class PipeLoader:
    """Loads the image model in the background while the cut-out / pose / face parse run."""
    def __init__(self):
        import threading
        self.pipe = None
        self.thread = None
        if os.environ.get("CHARACTER_EDIT_VARIANTS", "1") == "0": return
        self.thread = threading.Thread(target=self._load, daemon=True); self.thread.start()

    def _load(self):
        try: self.pipe = load_edit_pipe()
        except Exception as e: log(f"image model unavailable ({str(e)[:160]}) — using local shapes")

    def get(self, timeout=240):
        if self.thread: self.thread.join(timeout)
        return self.pipe


def clone_region(base_rgb, patch_rgb, region):
    """Poisson-blend `patch_rgb` into `base_rgb` inside `region` (uint8 arrays, same size)."""
    import numpy as np, cv2
    m = (region > 127).astype(np.uint8) * 255
    m[:2, :] = 0; m[-2:, :] = 0; m[:, :2] = 0; m[:, -2:] = 0
    if m.max() == 0: return base_rgb
    try:
        x, y, w, h = cv2.boundingRect(m)
        return cv2.seamlessClone(patch_rgb, base_rgb, m, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
    except Exception:
        a = cv2.GaussianBlur(m, (0, 0), 2).astype(np.float32)[..., None] / 255.0
        return (patch_rgb * a + base_rgb * (1 - a)).astype(np.uint8)


class InpaintEdits:
    """Batched in-place repaints. add() jobs, run(pipe) once, get(name) → blended full image."""
    def __init__(self, img, seed=1):
        self.img = img; self.seed = seed; self.jobs = {}; self.raw = {}

    def add(self, name, box, prompt, region, strength=0.85, min_change=0.0, change_mask=None, blend="clone"):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        W, H = self.img.size
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 16 or y1 - y0 < 16 or not has(region): return
        self.jobs[name] = {"box": (x0, y0, x1, y1), "prompt": prompt, "region": region, "strength": float(strength),
                           "min_change": min_change, "change_mask": change_mask, "blend": blend}

    def run(self, pipe):
        if pipe is None or not self.jobs: return
        import contextlib
        try: import torch
        except Exception: torch = None
        from PIL import Image, ImageFilter
        lcm = "lcm" in EDIT_MODEL.lower()
        groups = {}
        for name, j in self.jobs.items(): groups.setdefault((j["box"], round(j["strength"], 2)), []).append(name)
        t0 = time.time()
        for (box, strength), names in groups.items():
            crop = self.img.convert("RGB").crop(box)
            k = EDIT_SIZE / max(crop.size)
            w, h = max(64, int(crop.width * k) // 8 * 8), max(64, int(crop.height * k) // 8 * 8)
            src = crop.resize((w, h), Image.Resampling.LANCZOS)
            for i in range(0, len(names), max(1, EDIT_BATCH)):
                chunk = names[i:i + max(1, EDIT_BATCH)]
                masks = [self.jobs[n]["region"].crop(box).resize((w, h), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(1.5)) for n in chunk]
                steps = max(4, math.ceil(int(os.environ.get("CHARACTER_EDIT_STEPS", "4")) / max(0.25, strength))) if lcm else int(os.environ.get("CHARACTER_EDIT_STEPS", "14"))
                gens = [torch.Generator(device="cpu").manual_seed(int(self.seed) * 7 + sum(map(ord, n))) for n in chunk] if torch else None
                kw = dict(prompt=[self.jobs[n]["prompt"] for n in chunk], image=[src] * len(chunk), mask_image=masks,
                          strength=strength, num_inference_steps=steps, guidance_scale=1.0 if lcm else 6.5,
                          generator=gens, height=h, width=w)
                if not lcm: kw["negative_prompt"] = [EDIT_NEGATIVE] * len(chunk)
                t = time.time()
                try:
                    with (torch.inference_mode() if torch else contextlib.nullcontext()): outs = pipe(**kw).images
                except Exception as e:
                    log(f"variants {chunk} failed ({str(e)[:140]})"); continue
                for n, im in zip(chunk, outs): self.raw[n] = im.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
                log(f"painted {', '.join(chunk)} in {time.time() - t:.1f}s ({w}x{h}, {steps} steps)")
        log(f"all variants painted in {time.time() - t0:.1f}s")

    def get(self, name):
        """Full-size RGBA with the repaint blended in, or None if missing / rejected."""
        import numpy as np
        from PIL import Image
        if name not in self.raw: return None
        j = self.jobs[name]; x0, y0, x1, y1 = j["box"]
        base = np.array(self.img.convert("RGB").crop(j["box"]))
        out = np.array(self.raw[name])
        if j["min_change"] > 0 and j["change_mask"] is not None:
            cm = np.array(j["change_mask"].crop(j["box"])) > 127
            diff = np.abs(out.astype(np.int16) - base.astype(np.int16)).mean(axis=2)
            # Real change = change on the feature minus the model's overall colour drift
            # (measured on the untouched pixels outside the repaint).
            outside = np.array(j["region"].crop(j["box"])) < 20
            drift = float(diff[outside].mean()) if outside.any() else 0.0
            change = (float(diff[cm].mean()) if cm.any() else 0.0) - drift
            if change < j["min_change"]:
                log(f"variant {name}: rejected — the shape barely changed ({change:.1f})"); return None
        region = np.array(j["region"].crop(j["box"]))
        if j["blend"] == "clone":
            # Poisson blend: the repaint takes the lighting/colour of the skin around it.
            blended = clone_region(base, out, region)
        else:
            # Feathered paste (used when the border is not skin, e.g. the scalp under hair).
            import cv2
            al = cv2.GaussianBlur((region > 127).astype(np.float32), (0, 0), 1.5)[..., None]
            blended = (out * al + base * (1 - al)).astype(np.uint8)
        full = self.img.convert("RGBA").copy()
        full.paste(Image.fromarray(blended).convert("RGBA"), (x0, y0))
        return full

    def close(self):
        self.raw.clear()


def skin_like(img, region, ref_mask, tol=26.0):
    """Pixels in `region` whose Lab colour is close to the reference skin colour."""
    try:
        import cv2, numpy as np
        from PIL import Image
        if not has(region) or not has(ref_mask): return None
        lab = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref = lab[np.array(ref_mask) > 127]
        if len(ref) < 20: return None
        med = np.median(ref, axis=0)
        dist = np.sqrt(((lab[..., 1:] - med[1:]) ** 2).sum(axis=2) + ((lab[..., 0] - med[0]) * 0.35) ** 2)
        m = (dist < tol) & (np.array(region) > 127)
        return Image.fromarray((m * 255).astype(np.uint8), "L")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature locator: every face gets eyes, brows and a mouth.
#
# The face parser was trained on photos and often finds nothing on 3D, cartoon or anime
# faces — and no mouth meant no mouth layer, so no visemes and a character that cannot
# talk. Each feature is now resolved through a chain of independent sources:
#   1. the face parser (best edges when it works),
#   2. 68 face landmarks from the whole-body pose model,
#   3. colour: lips / eyes / brows differ from the character's own skin colour,
#      searched where the landmarks or the face proportions say they must be,
#   4. fixed face proportions anchored to the located head (the Live2D slicing
#      approach) — so a talking mouth layer always exists.
# ---------------------------------------------------------------------------
def _poly_mask(size, pts):
    from PIL import Image, ImageDraw
    m = Image.new("L", size, 0)
    if len(pts) >= 3: ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in pts], fill=255)
    return m


def landmark_regions(face, size, alpha=None):
    """68 face points → {"box", "l_eye", "r_eye", "l_brow", "r_brow", "nose", "u_lip",
    "l_lip", "mouth", "eye_c": {l, r}, "mouth_c", "mouth_w", "eye_dist"} or None if the
    points do not look like a face."""
    import numpy as np
    if not face or len(face) < 68: return None
    P = np.array([(x, y) for x, y, _ in face], np.float32); S = np.array([c for _, _, c in face], np.float32)
    W, H = size
    if float(np.median(S[36:68])) < float(os.environ.get("CHARACTER_LANDMARK_MIN", "0.3")): return None
    re, le = P[36:42].mean(0), P[42:48].mean(0)          # image-left eye = subject's right
    mo, mi = P[48:60], P[60:68]
    mc = mo.mean(0)
    ed = float(np.hypot(*(le - re)))
    if ed < 6 or le[0] <= re[0]: return None
    if not (mc[1] > (re[1] + le[1]) / 2 + ed * 0.35): return None   # mouth well below the eyes
    if abs(mc[0] - (re[0] + le[0]) / 2) > ed * 0.8: return None
    if not (0 <= mc[0] < W and 0 <= mc[1] < H): return None
    if alpha is not None:
        try:
            if alpha.getpixel((int(mc[0]), int(mc[1]))) < 128: return None
        except Exception: pass
    jaw = P[0:17]; brows = P[17:27]
    x0, x1 = float(min(jaw[:, 0].min(), brows[:, 0].min())), float(max(jaw[:, 0].max(), brows[:, 0].max()))
    btop, chin = float(brows[:, 1].min()), float(jaw[:, 1].max())
    box = (x0, btop - (chin - btop) * 0.45, x1, chin)
    out = {"box": box, "eye_c": {"l": tuple(le), "r": tuple(re)}, "mouth_c": tuple(mc),
           "mouth_w": float(mo[:, 0].max() - mo[:, 0].min()), "eye_dist": ed}
    out["r_eye"], out["l_eye"] = _poly_mask(size, P[36:42]), _poly_mask(size, P[42:48])
    for key, rng in (("r_brow", P[17:22]), ("l_brow", P[22:27])):
        th = max(2.0, ed * 0.06)
        out[key] = _poly_mask(size, list(rng + [0, -th]) + list((rng + [0, th])[::-1]))
    out["nose"] = _poly_mask(size, list(P[31:36]) + [P[30]])
    upper = list(P[48:55]) + list(P[60:65][::-1])
    lower = list(P[54:60]) + [P[48], P[60]] + list(P[64:68][::-1])
    out["u_lip"], out["l_lip"] = _poly_mask(size, upper), _poly_mask(size, lower)
    out["mouth"] = _poly_mask(size, mi)
    return out


def _lab(rgb):
    import cv2, numpy as np
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _fill_holes(m):
    import cv2, numpy as np
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    if cs: cv2.drawContours(out, cs, -1, 1, -1)
    return out


def _colour_blob(img, zone, skin_lab, centre, want_w, thr, exclude=None, darker_only=False, merge_gap=0.8, accept=None):
    """Pixels inside `zone` whose colour differs from the skin, grouped around `centre`.
    Returns (mask uint8 full size, Lab image crop info) or None."""
    import numpy as np, cv2
    from PIL import Image
    if not has(zone): return None
    W, H = img.size
    x0, y0, x1, y1 = zone.getbbox()
    rgb = np.array(img.convert("RGB").crop((x0, y0, x1, y1)))
    lab = _lab(rgb)
    z = np.array(zone.crop((x0, y0, x1, y1))) > 127
    if exclude is not None and has(exclude): z &= ~(np.array(exclude.crop((x0, y0, x1, y1))) > 127)
    d = np.sqrt(((lab - skin_lab) ** 2).sum(2))
    cand = (d > thr) & z
    if darker_only: cand &= lab[..., 0] < skin_lab[0] - thr * 0.8
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k = max(3, int(want_w * 0.06)) | 1
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_m, st, ce = cv2.connectedComponentsWithStats(cand, 8)
    cx, cy = centre[0] - x0, centre[1] - y0
    best, bs = 0, None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < max(6, (want_w ** 2) * 0.006): continue
        dist = np.hypot((ce[i][0] - cx) / max(1, want_w), (ce[i][1] - cy) / max(1, want_w * 0.6))
        if dist > 1.1: continue
        if accept is not None and not accept(st[i, 2], st[i, 3]): continue
        sc = a * (1.2 - min(1.0, dist))
        if bs is None or sc > bs: best, bs = i, sc
    if not best: return None
    bx, by, bw, bh = st[best, 0], st[best, 1], st[best, 2], st[best, 3]
    keep = lab_m == best
    # Parts split by a gap (upper / lower lip around teeth, lashes above an eye white).
    for i in range(1, n):
        if i == best: continue
        ix, iy, iw, ih = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        if st[i, cv2.CC_STAT_AREA] < max(4, st[best, cv2.CC_STAT_AREA] * 0.08): continue
        overlap = min(bx + bw, ix + iw) - max(bx, ix)
        gap = max(iy - (by + bh), by - (iy + ih))
        if overlap > min(bw, iw) * 0.5 and gap <= merge_gap * max(bh, ih) and iw >= bw * 0.45 and bw >= iw * 0.45:
            keep |= lab_m == i
    keep = _fill_holes(keep)
    full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = keep
    return full, (x0, y0, lab, d)


def find_mouth(img, zone, skin_lab, centre, want_w, exclude=None):
    """Lips / mouth from colour → (u_lip, l_lip, inner) PIL masks or None."""
    import numpy as np, cv2
    from PIL import Image
    r = _colour_blob(img, zone, skin_lab, centre, want_w, 16.0, exclude, merge_gap=0.6,
                     accept=lambda w, h: w >= want_w * 0.25 and h <= w * 1.25)
    if r is None: return None
    core, (x0, y0, lab, d) = r
    ys, xs = np.where(core > 0)
    if len(xs) < 8: return None
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if not (want_w * 0.3 <= w <= want_w * 2.2) or h > w * 1.25 or h < 2: return None
    Hc, Wc = lab.shape[:2]
    c = core[y0:y0 + Hc, x0:x0 + Wc] > 0
    L = lab[..., 0]
    lipish = c & (d > 16)
    lip_lab = np.median(lab[lipish], axis=0) if lipish.sum() > 4 else np.median(lab[c], axis=0)
    dl = np.sqrt(((lab - lip_lab) ** 2).sum(2))
    teeth = c & (L > max(skin_lab[0], lip_lab[0]) + 8) & (np.abs(lab[..., 1] - 128) < 14) & (np.abs(lab[..., 2] - 128) < 18)
    dark = c & (L < lip_lab[0] - 22) & (dl > 26)
    inner = (teeth | dark).astype(np.uint8)
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & c
    if inner.sum() < c.sum() * 0.04: inner[:] = 0
    lips = c & (inner == 0)
    # Seam: middle of the opening, else the darkest row of the closed lips.
    if inner.sum():
        iy = np.where(inner)[0]; seam = int((iy.min() + iy.max()) / 2)
    else:
        rows = [(L[yy][c[yy]].mean() if c[yy].sum() > 2 else 255) for yy in range(Hc)]
        seam = int(np.argmin(rows))
    top = np.zeros_like(c); top[:seam] = True
    def full(a):
        out = np.zeros((img.height, img.width), np.uint8); out[y0:y0 + Hc, x0:x0 + Wc] = a.astype(np.uint8) * 255
        return Image.fromarray(out, "L")
    u, l = full(lips & top), full(lips & ~top)
    if not has(u) or not has(l):
        # One-piece lips (a line mouth): split at the middle.
        ly = np.where(lips)[0]
        if len(ly):
            mid = int((ly.min() + ly.max()) / 2); top[:] = False; top[:mid + 1] = True
            u, l = full(lips & top), full(lips & ~top)
    return u, l, (full(inner > 0) if inner.sum() else None)


def find_eye(img, zone, skin_lab, centre, want_w, exclude=None):
    import numpy as np
    from PIL import Image
    r = _colour_blob(img, zone, skin_lab, centre, want_w, 22.0, exclude, merge_gap=0.12,
                     accept=lambda w, h: h >= w * 0.22 and want_w * 0.3 <= w <= want_w * 2.0)
    if r is None: return None
    core = r[0]
    ys, xs = np.where(core > 0)
    if len(xs) < 8: return None
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if not (want_w * 0.35 <= w <= want_w * 2.0): return None
    if h < w * 0.22: return None          # a lash line: the eye is closed in the picture
    return Image.fromarray(core * 255, "L")


def find_brow(img, zone, skin_lab, centre, want_w, exclude=None):
    import numpy as np
    from PIL import Image
    r = _colour_blob(img, zone, skin_lab, centre, want_w, 20.0, exclude, darker_only=True, merge_gap=0.3,
                     accept=lambda w, h: w >= want_w * 0.35 and h <= w * 0.7)
    if r is None: return None
    core = r[0]
    ys, xs = np.where(core > 0)
    if len(xs) < 6: return None
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if w < want_w * 0.4 or h > w * 0.7: return None
    return Image.fromarray(core * 255, "L")


def locate_features(img, m, alpha, pose, face):
    """Complete the parser's eye / brow / mouth masks in place. Returns (face_box or None,
    {feature: source}) — the box is the located head (forehead to chin)."""
    import numpy as np
    W, H = img.size
    src = {}
    feats = union(m["l_brow"], m["r_brow"], m["l_eye"], m["r_eye"], m["nose"], m["mouth"], m["u_lip"], m["l_lip"])
    lm = landmark_regions(face, (W, H), alpha)
    # --- the head box -------------------------------------------------------------
    box = None
    pf = bbox(union(m["skin"], feats)) if has(m["skin"]) or has(feats) else None
    if lm: box = lm["box"]
    elif pose and ok(pose, "l_eye", 0.3, (W, H)) and ok(pose, "r_eye", 0.3, (W, H)):
        le, re = pose["l_eye"][:2], pose["r_eye"][:2]
        ed = math.hypot(le[0] - re[0], le[1] - re[1])
        if ed > 6:
            fwid = ed * 2.4; ex, ey = (le[0] + re[0]) / 2, (le[1] + re[1]) / 2
            box = (ex - fwid / 2, ey - fwid * 0.62, ex + fwid / 2, ey + fwid * 0.78)
    if box is None and pf is not None and has(m["skin"]):
        box = pf
    guess = False
    if box is None:
        # Nothing found a face: the head is the top of the silhouette (the slicing-template
        # approach). Only skin-coloured spots get a mouth here.
        ab = alpha.getbbox() if has(alpha) else None
        if not ab: return None, src
        fw0 = (ab[2] - ab[0]) * 0.45
        box = ((ab[0] + ab[2]) / 2 - fw0 / 2, ab[1] + fw0 * 0.15, (ab[0] + ab[2]) / 2 + fw0 / 2, ab[1] + fw0 * 1.4)
        guess = True
    bx0, by0, bx1, by1 = box
    fw, fh = max(8.0, bx1 - bx0), max(8.0, by1 - by0)
    cx = (bx0 + bx1) / 2
    # --- anchors ----------------------------------------------------------------------
    nose_tip = None
    if lm:
        eye_c, ed = lm["eye_c"], lm["eye_dist"]
        nose_tip = float(face[33][1])
        mouth_c, mouth_w = lm["mouth_c"], max(lm["mouth_w"], ed * 0.45)
    else:
        if pose and ok(pose, "l_eye", 0.3, (W, H)) and ok(pose, "r_eye", 0.3, (W, H)):
            eye_c = {"l": pose["l_eye"][:2], "r": pose["r_eye"][:2]}
        else:
            eye_c = {"l": (cx + fw * .2, by0 + fh * .47), "r": (cx - fw * .2, by0 + fh * .47)}
        for s, k in (("l", "l_eye"), ("r", "r_eye")):
            if has(m[k]): b = m[k].getbbox(); eye_c[s] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        ed = max(6.0, math.hypot(eye_c["l"][0] - eye_c["r"][0], eye_c["l"][1] - eye_c["r"][1]))
        ey = (eye_c["l"][1] + eye_c["r"][1]) / 2; ex = (eye_c["l"][0] + eye_c["r"][0]) / 2
        if has(m["nose"]):
            nb = m["nose"].getbbox(); ny = nb[3]; nx = (nb[0] + nb[2]) / 2
            mouth_c = (nx, ny + (ny - ey) * 0.45); nose_tip = ny
        elif pose and ok(pose, "nose", 0.3, (W, H)) and pose["nose"][1] - ey > ed * 0.25:
            nx, ny = pose["nose"][:2]; mouth_c = (nx, ny + (ny - ey) * 0.75); nose_tip = ny + (ny - ey) * 0.2
        else:
            mouth_c = (ex, ey + ed * 0.95)
        mouth_w = ed * 0.75
    # Skin colour of THIS character: the face minus everything that is not skin.
    rgb = np.array(img.convert("RGB"))
    facez = intersect(shape_mask((W, H), "ellipse", [bx0 + fw * .1, by0 + fh * .25, bx1 - fw * .1, by1 - fh * .05]), alpha)
    not_skin = union(m["hair"], dilate(feats, 3), m["eyeglass"], m["hat"],
                     *[shape_mask((W, H), "ellipse", [x - ed * .35, y - ed * .25, x + ed * .35, y + ed * .25]) for x, y in eye_c.values()],
                     shape_mask((W, H), "ellipse", [mouth_c[0] - mouth_w * .8, mouth_c[1] - mouth_w * .4, mouth_c[0] + mouth_w * .8, mouth_c[1] + mouth_w * .4]))
    sk = subtract(intersect(facez, m["skin"]) if has(intersect(facez, m["skin"])) else facez, not_skin)
    px = rgb[np.array(sk) > 127] if has(sk) else rgb[np.array(facez) > 127] if has(facez) else None
    if px is None or len(px) < 20:
        return box, src
    lab_px = _lab(px.reshape(-1, 1, 3)).reshape(-1, 3)
    L = lab_px[:, 0]
    mid = (L >= np.percentile(L, 25)) & (L <= np.percentile(L, 85))
    skin_lab = np.median(lab_px[mid] if mid.sum() > 10 else lab_px, axis=0)
    hair_ex = m["hair"]
    # --- mouth ------------------------------------------------------------------------
    pm = union(m["mouth"], m["u_lip"], m["l_lip"])
    parser_ok = False
    if has(pm):
        b = pm.getbbox(); pc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); pw = b[2] - b[0]
        parser_ok = pw >= mouth_w * 0.35 and math.hypot(pc[0] - mouth_c[0], pc[1] - mouth_c[1]) < max(mouth_w, ed) * 0.9
        if not lm: parser_ok = parser_ok or pw >= ed * 0.25
    if parser_ok:
        src["mouth"] = "face parser"
    else:
        zone = shape_mask((W, H), "ellipse", [mouth_c[0] - mouth_w * 1.0, mouth_c[1] - mouth_w * .55, mouth_c[0] + mouth_w * 1.0, mouth_c[1] + mouth_w * .6])
        zone = subtract(intersect(zone, alpha), dilate(m["nose"], 2))
        if nose_tip is not None:   # never above the tip of the nose
            zone = subtract(zone, shape_mask((W, H), "rect", [0, 0, W, nose_tip]))
        found = find_mouth(img, zone, skin_lab, mouth_c, mouth_w, hair_ex)
        if found is None and lm and has(lm["u_lip"]):
            found = (lm["u_lip"], lm["l_lip"], lm["mouth"] if (np.array(lm["mouth"]) > 127).sum() > 12 else None)
            src["mouth"] = "face landmarks"
        elif found is not None:
            src["mouth"] = "colour (landmarks)" if lm else "colour (face proportions)"
        skin_at_mouth = True
        if found is None and guess:
            win = rgb[int(max(0, mouth_c[1] - 3)):int(mouth_c[1] + 4), int(max(0, mouth_c[0] - 3)):int(mouth_c[0] + 4)]
            skin_at_mouth = win.size > 0 and float(np.sqrt(((np.median(_lab(win).reshape(-1, 3), 0) - skin_lab) ** 2).sum())) < 22
        if found is None and skin_at_mouth:
            # Proportional fallback: a closed mouth where the face says it is.
            mw, mh = mouth_w * 0.9, max(4.0, mouth_w * 0.16)
            u = shape_mask((W, H), "ellipse", [mouth_c[0] - mw / 2, mouth_c[1] - mh, mouth_c[0] + mw / 2, mouth_c[1] + mh * 0.15])
            l = shape_mask((W, H), "ellipse", [mouth_c[0] - mw * .45, mouth_c[1] - mh * 0.1, mouth_c[0] + mw * .45, mouth_c[1] + mh * 1.1])
            found = (intersect(u, alpha), subtract(intersect(l, alpha), u), None)
            src["mouth"] = "face proportions"
        if found is not None:
            u, l, inner = found
            m["u_lip"], m["l_lip"] = u, l
            m["mouth"] = inner if inner is not None else empty_mask((W, H))
    # --- eyes and brows ------------------------------------------------------------------
    for s in ("l", "r"):
        k, bk = f"{s}_eye", f"{s}_brow"
        ex, ey = eye_c[s]
        want = ed * 0.55
        if has(m[k]):
            src[k] = "face parser"
        else:
            vy = 0.7 if lm else 1.0
            zone = intersect(shape_mask((W, H), "ellipse", [ex - want * .95, ey - want * vy, ex + want * .95, ey + want * vy]), alpha)
            e = find_eye(img, zone, skin_lab, (ex, ey), want, union(hair_ex, m[bk]))
            if e is None and lm and has(lm[k]):
                le = lm[k]; lb = le.getbbox()
                e = dilate(le, max(1, int((lb[3] - lb[1]) * 0.25)))
                src[k] = "face landmarks"
            elif e is not None:
                src[k] = "colour"
            if e is not None: m[k] = e
        if not has(m[bk]):
            eb = m[k].getbbox() if has(m[k]) else (ex - want / 2, ey - want * .3, ex + want / 2, ey + want * .3)
            bw = (eb[2] - eb[0]) * 1.15; byc = eb[1] - ed * 0.18
            zone = intersect(shape_mask((W, H), "ellipse", [ex - bw * .75, byc - ed * .2, ex + bw * .75, byc + ed * .12]), alpha)
            zone = subtract(zone, dilate(m[k], 2))
            b = find_brow(img, zone, skin_lab, (ex, byc), bw, hair_ex)
            if b is None and lm and has(lm[bk]):
                b = subtract(intersect(lm[bk], alpha), hair_ex); src[bk] = "face landmarks"
            elif b is not None: src[bk] = "colour"
            if b is not None and has(b): m[bk] = b
    if not has(m["nose"]) and lm and has(lm["nose"]):
        m["nose"] = lm["nose"]; src["nose"] = "face landmarks"
    if guess and not any(has(m[k]) for k in ("u_lip", "l_lip", "l_eye", "r_eye")):
        return None, src
    return box, src


def set_anchor(info_bbox, world_x, world_y):
    x0, y0, x1, y1 = info_bbox; w = max(1, x1 - x0); h = max(1, y1 - y0)
    return {"anchorX": max(0, min(100, ((world_x - x0) / w) * 100)), "anchorY": max(0, min(100, ((world_y - y0) / h) * 100))}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(input_path, out_dir, mode, plan, gender, full_body, seed):
    from PIL import Image, ImageChops
    import numpy as np
    out = Path(out_dir); parts_dir = out / "parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    src = load_pil(input_path)
    max_side = int(os.environ.get("CHARACTER_MAX_SIDE", "1024"))
    scale = min(1, max_side / max(src.size))
    if scale < 1: src = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))), Image.Resampling.LANCZOS)
    W, H = src.size
    log(f"source {W}x{H}, mode={mode}, requested fullBody={bool(full_body)}")
    loader = PipeLoader()   # the image model loads while the cut-out runs

    # 1) Subject matte. A transparent PNG already says what the character is.
    a_in = src.getchannel("A")
    transparent = int((np.array(a_in) < 128).sum())
    soft = None
    if transparent > W * H * 0.02:
        log("input has transparency — using its alpha as the character matte")
        soft = a_in
        img = Image.alpha_composite(Image.new("RGBA", src.size, (255, 255, 255, 255)), src)
    else:
        img = src.copy(); img.putalpha(255)

    # 2) Pose first: it also helps choose the best matte.
    pose = detect_pose(img)
    if soft is None: soft = subject_mask(img, pose)
    alpha = binarize(soft, 127) if soft is not None else None

    # 3) Face parsing.
    parser = FaceParser()
    labels, zoomed = parse_labels(parser, img, pose)
    m = {k: mask_for(labels, k) for k in PARTS if k != "background"}
    if alpha is None:
        alpha = dilate(union(*[m[k] for k in ("skin", "neck", "cloth", "hair", "l_ear", "r_ear", "hat")]), 1)
    alpha = main_components(binarize(alpha)) if has(alpha) else Image.new("L", (W, H), 255)
    # Soft edge (anti-aliased hair strands) limited to the kept components.
    soft = intersect(soft, dilate(alpha, 2)) if soft is not None else alpha
    for k in m: m[k] = intersect(m[k], dilate(alpha, 2))

    # 3b) Make sure every face has eyes, brows and a mouth (parser → landmarks → colour →
    # face proportions). Without a mouth there are no visemes and the character cannot talk.
    try:
        loc_box, feature_src = locate_features(img, m, alpha, pose, LANDMARKS.get("face"))
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"feature locator failed ({str(e)[:160]})"); loc_box, feature_src = None, {}
    for k in m: m[k] = intersect(m[k], dilate(alpha, 2)) if m[k] is not None else empty_mask((W, H))
    log(f"features: {feature_src or 'all from the face parser'}; landmarks={'yes' if LANDMARKS.get('face') else 'no'}")

    features = union(m["l_brow"], m["r_brow"], m["l_eye"], m["r_eye"], m["nose"], m["mouth"], m["u_lip"], m["l_lip"])
    face_all = union(m["skin"], features, m["l_ear"], m["r_ear"])
    hb = bbox(face_all) if has(features) else None
    if loc_box is not None:
        # The parser's skin may cover only part of a stylised face: the located head
        # (forehead to chin) keeps the head layer whole.
        lb = [int(max(0, loc_box[0])), int(max(0, loc_box[1])), int(min(W, loc_box[2])), int(min(H, loc_box[3]))]
        if hb is None: hb = tuple(lb)
        else:
            sk_area = (np.array(m["skin"]) > 127).sum()
            if sk_area < (lb[2] - lb[0]) * (lb[3] - lb[1]) * 0.45:
                hb = (min(hb[0], lb[0]), min(hb[1], lb[1]), max(hb[2], lb[2]), max(hb[3], lb[3]))
    face_found = hb is not None
    if not face_found:
        if pose and ok(pose, "nose", 0.3, (W, H)):
            nx, ny = pose["nose"][:2]; s = W * 0.12
            hb = (int(nx - s), int(ny - s * 1.2), int(nx + s), int(ny + s * 1.1))
        else:
            ab = alpha.getbbox() or (0, 0, W, H)
            fw0 = (ab[2] - ab[0]) * 0.45
            hb = (int((ab[0] + ab[2]) / 2 - fw0 / 2), ab[1], int((ab[0] + ab[2]) / 2 + fw0 / 2), int(ab[1] + fw0 * 1.25))
        log("no face features were parsed — the head is rigged as one piece")
    fw, fh = max(8, hb[2] - hb[0]), max(8, hb[3] - hb[1])
    fcx, fcy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
    sub_b = alpha.getbbox() or (0, 0, W, H)
    log(f"face box {hb}, subject box {sub_b}, zoomed parse={zoomed}")

    # Skin limited to the face (a parser can call arms/hands "skin").
    face_zone = shape_mask((W, H), "ellipse", [hb[0] - fw * .12, hb[1] - fh * .15, hb[2] + fw * .12, hb[3] + fh * .08])
    skin = intersect(m["skin"], face_zone)
    neck = subtract(m["neck"], face_zone) if has(m["neck"]) else None
    if neck is not None and not has(neck): neck = None
    hair = m["hair"]
    hat = m["hat"] if has(m["hat"]) else None

    # 4) Framing: portrait vs full body.
    subj_h = sub_b[3] - sub_b[1]
    if pose:
        ankles = ok(pose, "l_ankle", 0.35, (W, H)) or ok(pose, "r_ankle", 0.35, (W, H))
        knees = ok(pose, "l_knee", 0.4, (W, H)) or ok(pose, "r_knee", 0.4, (W, H))
        is_full = bool(ankles or (knees and subj_h >= fh * 4.6))
    else:
        is_full = subj_h >= fh * 5.2
    log(f"framing detected: {'full body' if is_full else 'portrait'} (requested {'full body' if full_body else 'portrait'})")

    # Shoulders.
    if pose and ok(pose, "l_shoulder", 0.3) and ok(pose, "r_shoulder", 0.3):
        shL, shR = pose["l_shoulder"][:2], pose["r_shoulder"][:2]
        sd = max(fw * 1.2, math.hypot(shL[0] - shR[0], shL[1] - shR[1]))
        shoulder_y = (shL[1] + shR[1]) / 2
    else:
        sd = fw * 1.9; shoulder_y = min(sub_b[3], hb[3] + fh * .2)
        shL, shR = (fcx + sd / 2, shoulder_y), (fcx - sd / 2, shoulder_y)
    if pose and ok(pose, "l_hip", 0.3) and ok(pose, "r_hip", 0.3):
        hip_y = (pose["l_hip"][1] + pose["r_hip"][1]) / 2; hip_x = (pose["l_hip"][0] + pose["r_hip"][0]) / 2
    else:
        hip_y = sub_b[1] + subj_h * (0.53 if is_full else 1.0); hip_x = fcx
    hip_y = min(hip_y, sub_b[3])

    # 5) Hair: the part over the skull moves with the head (front); long lengths stay behind.
    skull = shape_mask((W, H), "ellipse", [hb[0] - fw * .28, hb[1] - fh * .65, hb[2] + fw * .28, hb[3] + fh * .05])
    hair_front = intersect(hair, skull)
    # Strands that fall past the skull (over the ears, neck, shoulders, chest) hang from the
    # head: they move with the head and are drawn in front of the body.
    hair_back = subtract(hair, hair_front)
    scalp_region = shape_mask((W, H), "ellipse", [hb[0] - fw * .10, hb[1] - fh * .38, hb[2] + fw * .10, hb[1] + fh * .62])
    hair_reveal = intersect(dilate(hair, 4), scalp_region)

    # 6) Arms and hands (only when the whole arm through the hand is visible).
    arm_specs = []
    hands_visible = {"left": False, "right": False}
    head_block = dilate(union(face_all, hair_front, hat, neck), 3)
    # Limbs are rigged only on full-body pictures. In a portrait the "arms" are sleeves or
    # cropped shoulders: cutting them out leaves floating chunks, so they stay with the body
    # (which breathes and sways as one piece).
    if pose and is_full and os.environ.get("CHARACTER_RIG_ARMS", "1") != "0":
        for side_name, side, pre in (("left", 1, "l"), ("right", -1, "r")):
            S, E, Wr = pose.get(f"{pre}_shoulder"), pose.get(f"{pre}_elbow"), pose.get(f"{pre}_wrist")
            if not (ok(pose, f"{pre}_shoulder", 0.3, (W, H)) and ok(pose, f"{pre}_elbow", 0.3, (W, H)) and ok(pose, f"{pre}_wrist", 0.3, (W, H))):
                continue
            S, E, Wr = S[:2], E[:2], Wr[:2]
            if Wr[1] > H - H * 0.025 or Wr[0] < 2 or Wr[0] > W - 2: continue  # hand is cut by the frame
            if alpha.getpixel((int(min(W - 1, Wr[0])), int(min(H - 1, Wr[1])))) < 128 and not has(intersect(alpha, shape_mask((W, H), "ellipse", [Wr[0] - sd * .08, Wr[1] - sd * .08, Wr[0] + sd * .08, Wr[1] + sd * .08]))):
                continue
            L1, L2 = math.hypot(E[0] - S[0], E[1] - S[1]), math.hypot(Wr[0] - E[0], Wr[1] - E[1])
            if L1 < sd * 0.12 or L2 < sd * 0.1: continue
            dfx, dfy = (Wr[0] - E[0]) / L2, (Wr[1] - E[1]) / L2
            hl = sd * 0.34; hr = sd * 0.16
            hand_end = (Wr[0] + dfx * hl, Wr[1] + dfy * hl)
            upper = intersect(capsule((W, H), S, E, max(4, sd * 0.13)), alpha)
            fore = intersect(capsule((W, H), E, Wr, max(4, sd * 0.11)), alpha)
            hand_zone = intersect(capsule((W, H), Wr, (Wr[0] + dfx * hl * .7, Wr[1] + dfy * hl * .7), hr), alpha)
            hand = skin_like(img, hand_zone, skin)
            if has(hand):
                hand = largest_component(dilate(erode(hand, 1), 2), (Wr[0] + dfx * hl * .35, Wr[1] + dfy * hl * .35))
                hand = intersect(hand, hand_zone)
            hb_area = (np.array(hand) > 127).sum() if has(hand) else 0
            if hb_area < (hr * hr) * 0.6:   # gloves / no skin match → geometric hand
                hand = hand_zone
            upper, fore, hand = (subtract(x, head_block) for x in (upper, fore, hand))
            if not (has(upper) and has(fore) and has(hand)): continue
            hands_visible[side_name] = True
            arm_specs.append({"side": side, "pre": pre, "S": S, "E": E, "Wr": Wr, "end": hand_end,
                              "upper": upper, "fore": fore, "hand": hand, "hl": hl, "L1": L1, "L2": L2})
    log(f"hands visible: {hands_visible}; rigged arms: {[a['pre'] for a in arm_specs]}")
    arms_all = union(*[union(a["upper"], a["fore"], a["hand"]) for a in arm_specs]) if arm_specs else None

    # 7) Repairs and lifelike variants (the app's DreamShaper image model, in place).
    filled = img.copy()
    hair_fill_prompt = str((plan or {}).get("hair_fill_prompt") or "natural scalp and forehead skin, same skin tone and lighting")
    eye_parts = {}
    for side, key in (("l", "l_eye"), ("r", "r_eye")):
        eye_parts[side] = split_eye_details(m[key], img) if has(m[key]) else (None, None, None, None)
    # Eyes the parser missed (closed in the source, stylised art, sunglasses) still get a
    # blink layer, placed from the pose eye keypoints or the face proportions.
    blink_only = {}
    for side, key, pk, sx in (("l", "l_eye", "l_eye", 1), ("r", "r_eye", "r_eye", -1)):
        if has(m[key]) or not face_found: continue
        if pose and ok(pose, pk, 0.4, (W, H)): ex, ey = pose[pk][:2]
        else: ex, ey = fcx + sx * fw * .2, fcy - fh * .05
        blink_only[side] = intersect(shape_mask((W, H), "ellipse", [ex - fw * .11, ey - fh * .05, ex + fw * .11, ey + fh * .05]), face_zone)
    eyes_union = union(m["l_eye"], m["r_eye"], *blink_only.values())
    lips = union(m["u_lip"], m["l_lip"])
    inner = subtract(m["mouth"], lips) if has(m["mouth"]) else None
    mouth_core = union(m["mouth"], lips)
    mouth_mask = None
    # Open in the picture (teeth showing)? Then the image model also paints a closed mouth.
    src_mouth_open = has(inner) and has(lips) and (inner.getbbox()[3] - inner.getbbox()[1]) > max(2, (lips.getbbox()[3] - lips.getbbox()[1]) * 0.18)
    viseme_prompts = dict(VISEME_PROMPTS)
    if src_mouth_open:
        viseme_prompts["MBP"] = ("mouth closed, lips gently pressed together, no teeth visible", 0.92)
    if has(mouth_core):
        mb = mouth_core.getbbox()
        lw0, lh0 = mb[2] - mb[0], max(3, mb[3] - mb[1])
        room = shape_mask((W, H), "ellipse", [mb[0] - lw0 * .28, mb[1] - lh0 * .32, mb[2] + lw0 * .28, mb[3] + lh0 * .95])
        mouth_mask = union(dilate(mouth_core, max(3, int(lh0 * 0.3))), room)
        mouth_mask = subtract(intersect(intersect(mouth_mask, face_zone), dilate(alpha, 1)), dilate(m["nose"], 2))
        mouth_mask = mouth_mask.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(max(1.0, lh0 * 0.12)))

    who = "man" if gender == "male" else "woman"
    style = str((plan or {}).get("render_prompt") or "")[:160]
    style = (style + ", ") if style else ""
    edits = InpaintEdits(img, seed)
    if face_found and has(mouth_mask):
        mb2 = mouth_core.getbbox(); mside = max(lw0 * 2.6, lh0 * 4.5, fw * 0.75)
        mcx, mcy = (mb2[0] + mb2[2]) / 2, (mb2[1] + mb2[3]) / 2 + lh0 * 0.3
        mbox = [mcx - mside / 2, mcy - mside / 2, mcx + mside / 2, mcy + mside / 2]
        mreg = binarize(mouth_mask, 40)
        lips_change = dilate(mouth_core, 2)
        for shape, (desc, strength) in viseme_prompts.items():
            edits.add(f"mouth_{shape}", mbox, f"{style}close-up of a {who}'s face, {desc}, same lips, same skin and lighting, natural detailed teeth, seamless",
                      mreg, strength, min_change=0.0 if shape in ("CONS", "FV") else 5.0, change_mask=lips_change)
    if face_found and has(eyes_union):
        eb = eyes_union.getbbox(); ew, ehh = eb[2] - eb[0], max(3, eb[3] - eb[1])
        eside = max(ew * 1.5, fw * 0.9)
        ebox = [(eb[0] + eb[2]) / 2 - eside / 2, (eb[1] + eb[3]) / 2 - eside * 0.4, (eb[0] + eb[2]) / 2 + eside / 2, (eb[1] + eb[3]) / 2 + eside * 0.4]
        ereg = dilate(eyes_union, max(3, int(ehh * 0.25)))
        edits.add("eyes", ebox, f"{style}close-up of a {who}'s face, both eyes gently closed, relaxed closed eyelids with eyelashes, natural crease, same skin and lighting",
                  ereg, 0.85, min_change=4.0, change_mask=eyes_union)
    if has(hair_reveal):
        hrb = hair_reveal.getbbox()
        edits.add("scalp", [hrb[0] - 8, hrb[1] - 8, hrb[2] + 8, hrb[3] + 8], f"{style}{hair_fill_prompt}", hair_reveal, 1.0, blend="paste")
    for a in arm_specs:
        hc = ((a["Wr"][0] + a["end"][0]) / 2, (a["Wr"][1] + a["end"][1]) / 2); hs = a["hl"] * 1.3
        hreg = dilate(a["hand"], max(4, int(a["hl"] * .25)))
        for k, desc in HAND_PROMPTS.items():
            edits.add(f"hand_{a['pre']}_{k}", [hc[0] - hs, hc[1] - hs, hc[0] + hs, hc[1] + hs], f"{style}{who}'s {desc}, same skin tone, same sleeve and lighting, natural fingers",
                      hreg, 0.95, min_change=4.0, change_mask=a["hand"])
    pipe = loader.get() if edits.jobs else None
    edits.run(pipe)
    scalp = edits.get("scalp")
    filled = scalp if scalp is not None else (cv_inpaint(filled, hair_reveal, 7) if has(hair_reveal) else filled)
    hand_variants = {}
    mouth_open = mouth_round = None
    eye_closed = None

    # Clean background plate: the scene with the character removed (for re-staging the
    # animated character over its own background). Fast CPU fill (seconds).
    background = None
    if os.environ.get("CHARACTER_BACKGROUND_PLATE", "1") != "0" and transparent <= W * H * 0.02:
        hole = dilate(alpha, max(6, int(min(W, H) * 0.012)))
        if background is None:
            # Telea is slow on big holes: fill at 1/4 size, upscale, then keep the original outside.
            small = img.convert("RGB").resize((max(1, W // 4), max(1, H // 4)), Image.Resampling.BILINEAR)
            sh = hole.resize(small.size, Image.Resampling.NEAREST)
            fill = cv_inpaint(small.convert("RGBA"), sh, 9).resize((W, H), Image.Resampling.BICUBIC)
            background = img.copy(); background.paste(fill, (0, 0), hole)

    if pipe is not None:
        try:
            import gc; del pipe; pipe = None; loader.pipe = None; gc.collect(); log(f"image model released, {rss_mb()} MB RAM")
        except Exception: pass

    # Feature-free face for the head layer: brows/eyes/nose/mouth live on their own
    # layers, so the base underneath is skin (no doubled features when they move).
    face_clean = cv_inpaint(filled, dilate(features, max(2, int(fw * 0.015))), 5) if has(features) else filled

    # Arms that cross the torso leave a filled shirt behind when they move.
    body_src = filled
    torso_box = shape_mask((W, H), "rect", [min(shL[0], shR[0]) - sd * .05, shoulder_y - sd * .1, max(shL[0], shR[0]) + sd * .05, hip_y + (sd * .1 if is_full else H)])
    arms_over_torso = intersect(intersect(arms_all, torso_box), alpha) if arms_all is not None else None
    if has(arms_over_torso):
        body_src = cv_inpaint(filled, dilate(arms_over_torso, 2), 9)

    # 8) Masks for the static layers.
    head_base = union(skin, features, m["l_ear"], m["r_ear"], m["earring"], hair_reveal,
                      subtract(subtract(intersect(alpha, shape_mask((W, H), "ellipse", [hb[0] - fw * .06, hb[1] - fh * .06, hb[2] + fw * .06, hb[3] + fh * .02])), hair), neck))
    if not face_found:
        head_base = subtract(intersect(alpha, shape_mask((W, H), "ellipse", list(hb))), hair_back)
    head_base = subtract(head_base, arms_all)
    body_core = subtract(subtract(alpha, union(head_base, hair, hat, neck, m["eyeglass"])), arms_all)
    # Hair hanging over the body leaves holes in it; when the hair moves those holes would
    # show the background (the dark jagged strip). Keep the body solid behind the hair:
    # every hair pixel enclosed left-and-right by body on its row belongs to the body too,
    # painted from the clothes / skin around it.
    behind_hair = None
    if has(body_core) and has(hair_back):
        bc = np.array(body_core) > 127; hb_arr = np.array(dilate(hair_back, 2)) > 127
        cols = np.arange(W)[None, :]
        anyb = bc.any(axis=1)
        left = np.where(anyb, np.argmax(bc, axis=1), W); right = np.where(anyb, W - 1 - np.argmax(bc[:, ::-1], axis=1), -1)
        inside = (cols >= left[:, None]) & (cols <= right[:, None])
        enclosed = hb_arr & inside & (np.array(alpha) > 127)
        if enclosed.any(): behind_hair = Image.fromarray((enclosed * 255).astype(np.uint8), "L")
    body_mask = union(body_core, behind_hair, arms_over_torso)
    if has(behind_hair):
        body_src = cv_inpaint(body_src, dilate(behind_hair, 2), 9)
    if not has(body_mask): body_mask = m["cloth"] if has(m["cloth"]) else None

    assets = {}

    def soft_edge(mask):
        """Layer mask × the soft matte: interior unchanged, the outer silhouette anti-aliased."""
        from PIL import ImageChops
        return ImageChops.multiply(mask, ImageChops.lighter(soft, erode(alpha, 2))) if soft is not None else mask

    def save(pid, label, mask, source, parent, tags, mask_is_final=False):
        if not has(mask): return None
        info = extract(source, soft_edge(mask), parts_dir / f"{pid}.png")
        if info:
            info.update({"label": label, "parent": parent, "tags": tags})
            assets[pid] = info
        return info

    # Torso / legs.
    legs_info = None
    if is_full and has(body_mask):
        ov = max(6, int(subj_h * 0.035))
        torso_mask = intersect(body_mask, shape_mask((W, H), "rect", [0, 0, W, hip_y + ov]))
        legs_mask = intersect(body_mask, shape_mask((W, H), "rect", [0, hip_y - ov, W, H]))
        save("torso", "Torso", torso_mask, body_src, "root", ["Body", "Torso"])
        legs_info = save("legs", "Legs", legs_mask, body_src, "root", ["Body", "Legs"])
    else:
        save("torso", "Torso", body_mask, body_src, "root", ["Body", "Torso"])
    # The neck continues up under the chin (hidden at rest) so a head tilt never opens a gap.
    neck_src, neck_full = filled, neck
    if has(neck) and has(hair_back):
        # Strands across the neck: the neck stays whole behind them.
        nbx = neck.getbbox()
        col = intersect(dilate(hair_back, 2), shape_mask((W, H), "rect", [nbx[0], nbx[1], nbx[2], nbx[3]]))
        if has(col):
            neck = union(neck, col); filled = cv_inpaint(filled, dilate(col, 2), 7)
    if has(neck):
        nb = neck.getbbox(); nw = nb[2] - nb[0]
        under_chin = intersect(shape_mask((W, H), "rect", [nb[0] + nw * .08, max(0, nb[1] - fh * .3), nb[2] - nw * .08, nb[1] + 2]), dilate(face_zone, 4))
        if has(under_chin):
            neck_full = union(neck, under_chin)
            neck_src = cv_inpaint(filled, subtract(under_chin, neck), 7)
    save("neck", "Neck", neck_full, neck_src, "root", ["Neck", "Body"])
    save("hair_locks", "Hair Locks", hair_back, img, "headGroup", ["Hair"])
    save("head", "Head", head_base, face_clean, "headGroup", ["Head", "Skin"])
    save("hair_front", "Front Hair", hair_front, img, "headGroup", ["Hair"])
    save("hat", "Hat", hat, img, "headGroup", ["Accessory", "Hat"])
    save("brow_l", "Left Eyebrow", dilate(m["l_brow"], 1), img, "headGroup", ["Eyebrow"])
    save("brow_r", "Right Eyebrow", dilate(m["r_brow"], 1), img, "headGroup", ["Eyebrow"])
    eyes_open = []
    blink_ids = []
    if eye_closed is None:
        eye_closed = edits.get("eyes")
    gaze = []
    frame_ids = []
    for side in ("l", "r"):
        eye, iris, pupil, iris_src = eye_parts[side]
        if not has(eye): continue
        eye_src = sclera_fill(img, eye, intersect(iris, eye) if iris is not None else None)
        # Eye white: the whole opening (and a little under the lids for the moving iris).
        if save(f"eye_{side}", f"{side.upper()} Eyeball", dilate(eye, 2), eye_src, "headGroup", ["Eyeball"]): eyes_open.append(f"eye_{side}")
        if save(f"iris_{side}", f"{side.upper()} Iris", iris, iris_src or img, "headGroup", ["Iris"]): eyes_open.append(f"iris_{side}")
        if save(f"pupil_{side}", f"{side.upper()} Pupil", pupil, iris_src or img, "headGroup", ["Pupil"]): eyes_open.append(f"pupil_{side}")
        # Eyelid frame: lids, lashes and skin around the opening, drawn OVER the iris, so the
        # moving iris slides under the lids like a real eye instead of floating on the skin.
        eb = eye.getbbox(); ew, eh = eb[2] - eb[0], max(3, eb[3] - eb[1])
        ib = iris.getbbox() if iris is not None else None
        ir = ((ib[2] - ib[0]) / 2) if ib else eh * 0.5
        from PIL import ImageFilter as _IF
        # Soft outer edge (blends into the face when the head turns), hard inner edge (hides the iris).
        outer = dilate(eye, max(4, int(ir * 1.05))).filter(_IF.GaussianBlur(max(1.0, ir * 0.12)))
        ring = subtract(outer, eye)
        ring = subtract(ring, dilate(union(m["l_brow"], m["r_brow"]), 1))
        if save(f"lid_{side}", f"{side.upper()} Lid Frame", ring, img, "headGroup", ["Eyeball"]): frame_ids.append(f"lid_{side}")
        gaze.append(ew * 0.16)
        closed_src = eye_closed if eye_closed is not None else synth_blink(face_clean, img, eye)
        if closed_src is not None and save(f"blink_{side}", f"{side.upper()} Closed Eye", dilate(eye, 2), closed_src, "headGroup", ["Blink", "Eyelid"]):
            blink_ids.append(f"blink_{side}")
    for side, em in blink_only.items():
        closed_src = eye_closed if eye_closed is not None else synth_blink(face_clean, img, em)
        if closed_src is not None and save(f"blink_{side}", f"{side.upper()} Closed Eye", dilate(em, 2), closed_src, "headGroup", ["Blink", "Eyelid"]):
            blink_ids.append(f"blink_{side}")
    save("nose", "Nose", dilate(m["nose"], 1), img, "headGroup", ["Nose"])
    save("glasses", "Glasses", m["eyeglass"], img, "headGroup", ["Accessory", "Glasses"])

    # Mouth: ONE layer; every viseme is cut with the same mask so the swap is pixel-aligned.
    mouth_sets = {}
    visemes = {}
    viseme_source = "none"
    if has(mouth_mask) and save("mouth", "Mouth", mouth_mask, img, "headGroup", ["Mouth"]):
        teeth_hint = None
        for side in ("l", "r"):
            eye, iris, _, _ = eye_parts[side]
            if has(eye):
                try:
                    px = np.array(img.convert("RGB")).astype(np.float32)[(np.array(subtract(eye, dilate(iris, 1))) > 127)]
                    if len(px) > 8:
                        lum = px.mean(1); teeth_hint = np.median(px[lum >= np.percentile(lum, 60)], axis=0); break
                except Exception: pass
        rel_mouth = str(Path(assets["mouth"]["file"]).relative_to(out).as_posix())
        try:
            built = build_visemes(img, face_clean, m["u_lip"] if has(m["u_lip"]) else None, m["l_lip"] if has(m["l_lip"]) else None, inner, mouth_mask, teeth_hint)
        except Exception as e:
            log(f"lip-warp visemes failed ({str(e)[:160]})"); built = {}
        # Lifelike shapes repainted by the image model win. A rejected shape borrows the
        # closest good one; the lip-warp set is used only if the model could not run at all.
        real = {}
        for shape in viseme_prompts:
            im = edits.get(f"mouth_{shape}")
            if im is not None: real[shape] = im
        neutral = {"REST": rel_mouth}
        if real:
            files = {}
            for shape, im in real.items():
                info = extract(im, soft_edge(mouth_mask), parts_dir / f"mouth_{shape.lower()}.png")
                if info: files[shape] = str(Path(info["file"]).relative_to(out).as_posix())
            for shape in VISEME_SHAPES:
                if shape in neutral: continue
                if shape in files: neutral[shape] = files[shape]; continue
                for alt in VISEME_FALLBACK.get(shape, []):
                    if alt in files or alt == "REST":
                        neutral[shape] = files.get(alt, rel_mouth); break
                neutral.setdefault(shape, files.get("AI") or rel_mouth)
            # A grin can't close by borrowing REST: use the lip-warp closed mouth instead.
            if src_mouth_open and "MBP" not in files and built.get("neutral", {}).get("MBP") is not None:
                info = extract(built["neutral"]["MBP"], soft_edge(mouth_mask), parts_dir / "mouth_mbp.png")
                if info: neutral["MBP"] = str(Path(info["file"]).relative_to(out).as_posix())
            viseme_source = f"painted {len(files)}/{len(viseme_prompts)}"
            log(f"mouth shapes: painted {sorted(files)}; borrowed {sorted(k for k in VISEME_SHAPES if k not in files and k != 'REST')}")
        elif built.get("neutral"):
            for shape, im in built["neutral"].items():
                if shape == "REST": continue
                info = extract(im, soft_edge(mouth_mask), parts_dir / f"mouth_{shape.lower()}.png")
                if info: neutral[shape] = str(Path(info["file"]).relative_to(out).as_posix())
            viseme_source = "lip-warp (image model unavailable)"
        if len(neutral) > 1:
            # Same mouth in every mood: the engine's expression controls do the rest, so the
            # lips are never bent into shapes the character's face does not have.
            mouth_sets = {mood: dict(neutral) for mood in MOODS}
        # Optional diffusion shapes replace the open / round ones when explicitly enabled.
        for shape, im in (("AI", mouth_open), ("O", mouth_round)):
            if im is not None and "neutral" in mouth_sets:
                info = extract(im, soft_edge(mouth_mask), parts_dir / f"mouth_diffusion_{shape.lower()}.png")
                if info: mouth_sets["neutral"][shape] = str(Path(info["file"]).relative_to(out).as_posix()); viseme_source = "lip-warp+inpainted"
        if not mouth_sets:
            # Last resort: the original synthetic overlays.
            v = {"REST": rel_mouth}
            for k, key in (("open", "AI"), ("round", "O"), ("wide", "E")):
                im = synth_mouth(img, lips, inner, k)
                info = extract(im, soft_edge(mouth_mask), parts_dir / f"mouth_{k}.png") if im is not None else None
                if info: v[key] = str(Path(info["file"]).relative_to(out).as_posix())
            op, rnd, wide = v.get("AI", rel_mouth), v.get("O", rel_mouth), v.get("E", rel_mouth)
            mouth_sets = {"neutral": {"REST": rel_mouth, "MBP": rel_mouth, "AI": op, "E": wide, "O": rnd, "U": rnd, "L": op, "CONS": wide, "FV": wide}}
            viseme_source = "synthetic"
        for mood in MOODS:
            mouth_sets.setdefault(mood, mouth_sets["neutral"])
        visemes = mouth_sets["neutral"]

    # Arm layers (bone-aligned) + hand poses.
    arm_layers = []
    for a in arm_specs:
        n = a["pre"]
        up = bone_layer(img, soft_edge(a["upper"]), a["S"], a["E"], parts_dir / f"upper_arm_{n}.png")
        fo = bone_layer(img, soft_edge(a["fore"]), a["E"], a["Wr"], parts_dir / f"forearm_{n}.png")
        hands = {}
        relaxed = bone_layer(img, soft_edge(a["hand"]), a["Wr"], a["end"], parts_dir / f"hand_{n}_relaxed.png")
        if not (up and fo and relaxed): continue
        hands["relaxed"] = relaxed
        for k in ("open", "point", "fist"):
            v = hand_variants.get((n, k))
            if v is None: v = edits.get(f"hand_{n}_{k}")
            if v is not None: hand_variants[(n, k)] = v
            info = None
            if v is not None:
                zone = dilate(a["hand"], max(4, int(a["hl"] * .3)))
                vm = subject_mask(v, None, MATTE_MODELS[:1])
                vm = intersect(zone, vm) if vm is not None else intersect(zone, alpha)
                sk = skin_like(v, vm, skin)
                if has(sk) and (np.array(sk) > 127).sum() > (np.array(a["hand"]) > 127).sum() * 0.4:
                    vm = intersect(dilate(largest_component(sk, a["Wr"]), 2), vm)
                info = bone_layer(v, vm, a["Wr"], a["end"], parts_dir / f"hand_{n}_{k}.png") if has(vm) else None
            hands[k] = info or relaxed
        arm_layers.append((a, up, fo, hands))

    # 9) Manifest (units = source pixels, x centred on the canvas so 0 is the middle).
    CX = W / 2.0
    def part(pid, info, label, parent, tags, opacity=1):
        x0, y0, x1, y1 = info["bbox"]
        return {"id": pid, "label": label, "tags": tags, "file": info["file"], "width": x1 - x0, "height": y1 - y0,
                "transform": {"x": (x0 + x1) / 2 - CX, "y": (y0 + y1) / 2, "rotation": 0, "scaleX": 1, "scaleY": 1, "anchorX": 50, "anchorY": 50},
                "zIndex": 0, "parentId": parent, "children": [], "isGroup": False, "isIndependent": True, "isVisible": True, "opacity": opacity}

    comp = {
        "root": {"id": "root", "label": "AI Character", "imageUrl": None, "transform": {"x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "anchorX": 50, "anchorY": 50},
                 "zIndex": 0, "tags": [], "parentId": None, "children": [], "isGroup": True, "isIndependent": False, "isVisible": True},
        "headGroup": {"id": "headGroup", "label": "Head Group", "imageUrl": None,
                      "transform": {"x": fcx - CX, "y": fcy, "rotation": 0, "scaleX": 1, "scaleY": 1, "anchorX": 50, "anchorY": 50},
                      "width": fw, "height": fh, "zIndex": 0, "tags": ["Head"], "parentId": "root", "children": [], "isGroup": True, "isIndependent": False, "isVisible": True},
    }
    for pid, a in assets.items():
        comp[pid] = part(pid, a, a["label"], a["parent"], a["tags"], 0 if pid.startswith("blink_") else 1)

    # Pivots: head turns on the neck, the neck on the chest, the torso on the hips, legs on the feet.
    neck_top = (fcx, hb[3] - fh * 0.04)
    if "neck" in assets:
        nb = assets["neck"]["bbox"]; neck_top = ((nb[0] + nb[2]) / 2, nb[1] + (nb[3] - nb[1]) * 0.25)
        comp["neck"]["transform"].update(set_anchor(nb, (nb[0] + nb[2]) / 2, nb[3]))
    if "head" in assets: comp["head"]["transform"].update(set_anchor(assets["head"]["bbox"], neck_top[0], min(assets["head"]["bbox"][3], neck_top[1])))
    comp["headGroup"]["transform"].update(set_anchor(list(hb), neck_top[0], min(hb[3], neck_top[1])))
    if "torso" in assets:
        tb = assets["torso"]["bbox"]
        comp["torso"]["transform"].update(set_anchor(tb, hip_x, min(tb[3], hip_y) if is_full else tb[3]))
    if legs_info:
        lb = legs_info["bbox"]; comp["legs"]["transform"].update(set_anchor(lb, (lb[0] + lb[2]) / 2, lb[3]))

    arms_meta = []
    arm_ids = []
    for a, up, fo, hands in arm_layers:
        n, side = a["pre"], a["side"]
        def bone_part(pid, label, info, joint, tags, opacity=1):
            w, h = info["width"], info["height"]
            ox, oy = info["jx"] - w / 2, info["jy"] - h / 2
            comp[pid] = {"id": pid, "label": label, "tags": tags, "file": info["file"], "width": w, "height": h,
                         "transform": {"x": joint[0] - CX - ox, "y": joint[1] - oy, "rotation": 0, "scaleX": 1, "scaleY": 1,
                                       "anchorX": info["jx"] / w * 100, "anchorY": info["jy"] / h * 100},
                         "zIndex": 0, "parentId": "root", "children": [], "isGroup": False, "isIndependent": True, "isVisible": True, "opacity": opacity}
            return {"ox": ox, "oy": oy}
        anchors = {}
        uid, fid = f"upper_arm_{n}", f"forearm_{n}"
        anchors[uid] = bone_part(uid, f"Upper Arm {n.upper()}", up, a["S"], ["Arm"])
        anchors[fid] = bone_part(fid, f"Forearm {n.upper()}", fo, a["E"], ["Arm"])
        hand_ids = {}
        for k in HAND_POSES:
            hid = f"hand_{n}_{k}"
            anchors[hid] = bone_part(hid, f"Hand {n.upper()} {k}", hands[k], a["Wr"], ["Hand"], 1 if k == "relaxed" else 0)
            hand_ids[k] = hid
        arm_ids += [hand_ids[k] for k in HAND_POSES] + [fid, uid]
        arms_meta.append({"side": side, "shoulder": {"x": a["S"][0] - CX, "y": a["S"][1]}, "upper": uid, "fore": fid,
                          "hands": hand_ids, "anchors": anchors, "upperLen": up["length"], "foreLen": fo["length"]})

    # Draw order: the engine paints the FIRST child on top. Front → back.
    head_order = ["glasses", "hat", "hair_front", "brow_l", "brow_r", *blink_ids, *frame_ids, "pupil_l", "pupil_r", "iris_l", "iris_r",
                  "eye_l", "eye_r", "nose", "mouth", "head", "hair_locks"]
    comp["headGroup"]["children"] = [p for p in head_order if p in comp]
    root_order = arm_ids + ["headGroup", "neck", "torso", "legs", "hair_back"]
    comp["root"]["children"] = [p for p in root_order if p in comp]
    for pid, p in comp.items():
        if pid in ("root", "headGroup"): continue
        if pid not in comp["root"]["children"] and pid not in comp["headGroup"]["children"]:
            comp[p["parentId"]]["children"].append(pid)

    top, feet = float(sub_b[1]), float(sub_b[3])
    if is_full:
        pelvis = float(hip_y); waist = pelvis - subj_h * 0.05
    else:
        pelvis = feet; waist = max(float(shoulder_y) + fh * 0.5, feet - 90.0)
    up_len = sum(a["L1"] for a in arm_specs) / len(arm_specs) if arm_specs else fh * .68
    fo_len = sum(a["L2"] for a in arm_specs) / len(arm_specs) if arm_specs else fh * .62
    geom = {
        "frame": {"top": top, "pelvis": pelvis, "waist": waist, "feet": feet},
        "headC": {"x": fcx - CX, "y": fcy}, "eyeY": fcy - fh * .12, "browY": fcy - fh * .24, "noseY": fcy + fh * .06,
        "mouthY": fcy + fh * .24, "earY": fcy, "faceW": float(fw), "faceH": float(fh),
        "eyeDX": fw * .21, "eyeW": fw * .18, "eyeH": fh * .13, "irisR": max(3.0, fw * .04),
        "shoulderY": float(shoulder_y), "shoulderX": float(sd / 2), "torsoW": float(sd * 1.05), "torsoH": float(max(fh, pelvis - shoulder_y)),
        "gazeRange": float(sum(gaze) / len(gaze)) if gaze else float(fw * .03),
        "upperW": sd * .13, "foreW": sd * .11, "upperLen": float(up_len), "foreLen": float(fo_len), "handLen": float(sd * .34),
    }
    name = str((plan or {}).get("name") or ("Generated Presenter" if mode == "generate" else "Imported Character"))[:80]
    manifest = {
        "version": 2, "origin": "center", "kind": "ai-rig", "name": name, "gender": gender, "fullBody": bool(is_full),
        "requestedFullBody": bool(full_body), "sourceMode": mode,
        "sourcePrompt": str((plan or {}).get("prompt") or "")[:1200], "canvas": {"width": W, "height": H}, "geometry": geom,
        "characters": [{"id": "presenter", "name": name, "composition": comp}], "visemeMap": visemes,
        "mouthSets": mouth_sets if mouth_sets else {"neutral": visemes, "smile": visemes, "happy": visemes, "sad": visemes},
        "eyes": {"happy": [], "open": [x for x in eyes_open if x in comp]}, "blink": blink_ids,
        "extras": {"tears": [], "glow": ""}, "arms": arms_meta,
        "camera": {"x": 0, "y": 0, "scale": 1, "rotation": 0}, "filters": {}, "lights": [], "ambient": 1, "aspect": {"w": 9, "h": 16},
        "detected": {"fullBody": bool(is_full), "handsVisible": hands_visible, "pose": bool(pose), "faceFound": face_found,
                     "zoomedFaceParse": zoomed, "handPoses": sorted({k for (_, k) in hand_variants}),
                     "blink": "painted" if eye_closed is not None else ("synthetic" if blink_ids else "none"),
                     "visemes": viseme_source, "features": feature_src, "landmarks": bool(LANDMARKS.get("face"))},
        "notes": {"faceParser": "BiSeNet/CelebAMask-HQ 19-class", "pose": "RTMW whole-body (rtmlib) with 68 face landmarks", "inpaint": "DreamShaper-8 inpainting with CPU fallbacks"},
    }
    if background is not None:
        bg = background.convert("RGB")
        if max(bg.size) > 1280:
            r = 1280.0 / max(bg.size); bg = bg.resize((int(bg.width * r), int(bg.height * r)), Image.Resampling.LANCZOS)
        # Every artifact PNG must stay under the app's 1.9 MB asset limit — shrink until it fits.
        for _ in range(8):
            bg.save(out / "background.png", "PNG", optimize=True, compress_level=9)
            if (out / "background.png").stat().st_size <= 1_700_000: break
            bg = bg.resize((max(64, int(bg.width * .8)), max(64, int(bg.height * .8))), Image.Resampling.LANCZOS)
        if (out / "background.png").stat().st_size <= 1_700_000:
            manifest["background"] = {"file": "background.png", "width": W, "height": H}
        else:
            (out / "background.png").unlink()
    edits.close()
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf8"), indent=2)
    preview = filled.copy(); preview.putalpha(alpha)
    if max(preview.size) > 768:
        r = 768.0 / max(preview.size); preview = preview.resize((max(1, int(preview.width * r)), max(1, int(preview.height * r))), Image.Resampling.LANCZOS)
    preview.save(out / "preview.png", "PNG", optimize=True, compress_level=9)
    try:
        # What was found where (opens from the run's artifact): eyes cyan, brows yellow,
        # lips magenta, mouth inside red, hair green, face box white.
        dbg = img.convert("RGB").copy()
        tint = [(m["hair"], (40, 220, 90)), (union(m["l_eye"], m["r_eye"]), (0, 220, 255)), (union(m["l_brow"], m["r_brow"]), (255, 220, 0)),
                (union(m["u_lip"], m["l_lip"]), (255, 0, 200)), (m["mouth"], (255, 40, 40)), (m["nose"], (255, 140, 0))]
        for mk, col in tint:
            if has(mk): dbg.paste(Image.blend(dbg, Image.new("RGB", dbg.size, col), 0.55), (0, 0), mk)
        from PIL import ImageDraw as _ID
        _ID.Draw(dbg).rectangle(list(hb), outline=(255, 255, 255), width=max(2, W // 300))
        r = 640.0 / max(dbg.size)
        if r < 1: dbg = dbg.resize((int(dbg.width * r), int(dbg.height * r)), Image.Resampling.LANCZOS)
        dbg.save(out / "debug_features.png", "PNG", optimize=True)
    except Exception as e:
        log(f"debug sheet skipped ({str(e)[:80]})")
    log(f"rig ready: {len([p for p in comp.values() if not p.get('isGroup')])} layers, arms={len(arms_meta)}, blink={manifest['detected']['blink']}, visemes={manifest['detected']['visemes']}, fullBody={is_full}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["generate", "upload"], default="upload"); ap.add_argument("--gender", default="female")
    ap.add_argument("--full-body", action="store_true"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--plan", default="{}")
    args = ap.parse_args()
    try:
        try: plan = json.loads(args.plan) if args.plan else {}
        except Exception: plan = {}
        build(args.input, args.output, args.mode, plan, args.gender, args.full_body, args.seed)
        return 0
    except Exception:
        traceback.print_exc(); return 1


if __name__ == "__main__": sys.exit(main())
