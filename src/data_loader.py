"""Phase 1+2 - Data collection understanding, preprocessing and EDA.

Finds real images (TrashNet / Garbage-Classification / manual folders),
removes corrupted + duplicate files, checks class distribution,
and produces stratified 70/20/10 splits. Falls back to synthetic
images so the pipeline runs without the dataset.
"""
import hashlib
import os
import random
from collections import Counter

from PIL import Image, ImageDraw

from .config import CLASSES, DATA_CANDIDATES, SEED, SPLITS

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_dataset_root(explicit=None):
    if explicit and os.path.isdir(explicit):
        return explicit
    for p in DATA_CANDIDATES:
        if os.path.isdir(p):
            # accept if it (or a subdir) holds class folders / images
            if has_images(p):
                return p
    return None


def has_images(root):
    for _dp, _dn, fns in os.walk(root):
        if any(f.lower().endswith(IMG_EXTS) for f in fns):
            return True
    return False


def _file_hash(path, n=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def scan_dataset(root):
    """Walk class-wise folders. Returns [(path, class_id)]."""
    items = []
    for cls in CLASSES:
        for cand in (cls, cls.capitalize(), cls.upper()):
            d = os.path.join(root, cand)
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith(IMG_EXTS):
                        items.append((os.path.join(d, fn), cand.lower()))
    # also support flat layout with class prefix e.g. img_plastic_12.jpg
    if not items:
        for fn in sorted(os.listdir(root)):
            if fn.lower().endswith(IMG_EXTS):
                low = fn.lower()
                for cls in CLASSES:
                    if cls in low:
                        items.append((os.path.join(root, fn), cls))
                        break
    return items


def clean_items(items):
    """Drop unreadable (corrupt) and duplicate (same content hash) files."""
    seen, good, dropped = set(), [], {"corrupt": 0, "duplicate": 0}
    for path, cid in items:
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            dropped["corrupt"] += 1
            continue
        h = _file_hash(path)
        if h in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(h)
        good.append((path, cid))
    return good, dropped


def stratified_split(items, splits=SPLITS, seed=SEED):
    """70/20/10 stratified on class. Returns dict split -> list."""
    rng = random.Random(seed)
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    out = {"train": [], "val": [], "test": []}
    for cls, lst in by_cls.items():
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(n * splits["train"])
        n_va = int(n * splits["val"])
        out["train"] += lst[:n_tr]
        out["val"] += lst[n_tr:n_tr + n_va]
        out["test"] += lst[n_tr + n_va:]
    for k in out:
        rng.shuffle(out[k])
    return out


def distribution(items):
    return dict(Counter(c for _, c in items))


def load_or_report(explicit=None):
    """Returns (splits_dict, info). Uses synthetic fallback if no data found."""
    root = find_dataset_root(explicit)
    if root is None:
        print("[data] No image dataset found under data/ - using synthetic fallback.")
        items = synthetic_items(n_per_class=30)
        return stratified_split(items), {"real": False, "root": None,
                                         "dist": distribution(items)}
    items = scan_dataset(root)
    items, dropped = clean_items(items)
    print(f"[data] root={root} kept={len(items)} dropped={dropped}")
    print(f"[data] class distribution: {distribution(items)}")
    return stratified_split(items), {"real": True, "root": root,
                                     "dist": distribution(items), "dropped": dropped}


# ---------------------------------------------------------------- synthetic
# Simple recognizable shapes per class so the pipeline is testable offline.
_SHAPE = {
    "plastic": ("ellipse", (60, 120, 200, 90)),
    "metal": ("rectangle", (255, 80, 60)),
    "paper": ("rectangle", (240, 240, 240)),
    "glass": ("triangle", (150, 200, 255)),
    "cardboard": ("rectangle", (180, 130, 70)),
    "trash": ("scribble", (100, 100, 100)),
}


def synthetic_image(cls, size=(224, 224), seed=0):
    rng = random.Random(hash((cls, seed)) % (2 ** 32))
    img = Image.new("RGB", size, (230, 230, 230))
    d = ImageDraw.Draw(img)
    kind, col = _SHAPE[cls]
    x0, y0 = rng.randint(20, 80), rng.randint(20, 80)
    x1, y1 = x0 + rng.randint(80, 140), y0 + rng.randint(80, 140)
    if kind == "ellipse":
        d.ellipse([x0, y0, x1, y1], fill=col)
    elif kind == "triangle":
        d.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=col)
    elif kind == "scribble":
        for _ in range(12):
            d.line([rng.randint(0, 223) for _ in range(4)], fill=col, width=3)
    else:
        d.rectangle([x0, y0, x1, y1], fill=col)
    return img


def synthetic_items(n_per_class=30, outdir=None):
    import tempfile
    outdir = outdir or tempfile.mkdtemp(prefix="waste_synth_")
    items = []
    for cls in CLASSES:
        d = os.path.join(outdir, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            p = os.path.join(d, f"syn_{i}.jpg")
            synthetic_image(cls, seed=i).save(p)
            items.append((p, cls))
    return items
