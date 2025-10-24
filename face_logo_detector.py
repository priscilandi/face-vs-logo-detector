import os
import glob
import cv2 
import sqlite3
import pandas as pd
import argparse 
from typing import Optional, Tuple         #which type of jpeg
import sys
from urllib.parse import urlsplit, unquote 
import numpy as np 
from PIL import Image 
from PIL import Image
from PIL import ImageDraw
import time

# --- Annotation output (framed faces) ---
ANNOTATE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "faces_marked")
LAST_FACE_RECTS = {}  # image_path -> list of (x, y, w, h)

def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


# Helper to clear all files/subfolders from a directory (used to reset annotation output)
def _clear_dir_contents(path: str):
    """Remove all files/subfolders under path. If path doesn't exist, create it."""
    try:
        if os.path.isdir(path):
            for name in os.listdir(path):
                fp = os.path.join(path, name)
                try:
                    if os.path.isfile(fp) or os.path.islink(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        import shutil
                        shutil.rmtree(fp)
                except Exception:
                    pass
        else:
            os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def save_annotated_face(image_path: str, rects, out_dir: str = ANNOTATE_OUTPUT_DIR, prefix: str = None):
    """Draw green frames using Pillow (ImageDraw) and save to out_dir with unique basename (optionally using prefix)."""
    try:
        if not rects:
            return
        _ensure_dir(out_dir)
        bgr = imread_any(image_path)
        if bgr is None:
            return
        # Convert BGR (cv2) -> RGB (Pillow)
        rgb = bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            font = None
        for (x, y, w, h) in rects:
            try:
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                label = "FACE"
                if font:
                    tw, th = draw.textsize(label, font=font)
                else:
                    tw, th = draw.textsize(label)
                pad = 2
                bx2, by2 = x1 + tw + 2*pad, y1 + th + 2*pad
                draw.rectangle([x1, y1, bx2, by2], fill=(0, 255, 0))
                draw.text((x1 + pad, y1 + pad), label, fill=(0, 0, 0), font=font if font else None)
            except Exception:
                continue
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        if not ext:
            ext = ".jpg"
        # Build a candidate name with optional prefix
        out_stem = f"{prefix}_{name}" if prefix else name
        out_path = os.path.join(out_dir, f"{out_stem}{ext}")
        # Avoid overwriting existing files by adding a numeric suffix if necessary
        if os.path.exists(out_path):
            k = 2
            while True:
                candidate = os.path.join(out_dir, f"{out_stem}_{k}{ext}")
                if not os.path.exists(candidate):
                    out_path = candidate
                    break
                k += 1
        try:
            pil_img.save(out_path)
        except Exception:
            pass
    except Exception:
        pass

# Variable opcional para probar con una sola imagen específica.
DEBUG_IMAGE_PATH = None  #define varaiable para usarla mas adelante

#  Importamos librerías necesarias: manejo de archivos, imágenes, bases de datos y utilidades.
# --- Robust OpenCV cascade directory resolution ---
#  Intentamos obtener la ruta donde OpenCV tiene almacenadas las cascadas HAAR.

try:
    # Some builds expose cv2.data.haarcascades
    from cv2 import data as cv2_data
    _CV2_HAAR_DIR = getattr(cv2_data, "haarcascades", None)
except Exception:
    _CV2_HAAR_DIR = None


def _existing_dir(p: str) -> bool:
    return isinstance(p, str) and os.path.isdir(p)


#buscar la carpeta donde opencv tiene las casacadas de haar 
 # Busca la ruta donde están las cascadas HAAR de OpenCV.
def get_haar_dir() -> Optional[str]:
    """Return a directory that contains OpenCV haarcascades.
    Tries multiple common locations across pip/conda installs.
    """
    candidates = []
    if _CV2_HAAR_DIR:
        candidates.append(_CV2_HAAR_DIR) #posibles rutas donde estan cascadas
    # Try relative to the cv2 package
    try:
        cv2_dir = os.path.dirname(cv2.__file__) 
        candidates.extend([
            os.path.join(cv2_dir, "data/haarcascades"), #rutas
            os.path.join(cv2_dir, "data"),    
            os.path.join(cv2_dir, "haarcascades"),
        ])
    except Exception:
        pass
    # Common conda/homebrew locations
    candidates.extend([
        "/opt/anaconda3/share/opencv4/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
        "/usr/share/opencv4/haarcascades",
    ])
    for c in candidates:
        if _existing_dir(c):
            return c if c.endswith(os.sep) else c + os.sep #recorremos todo y verificamos si se econtro
    return None


 # Busca la ruta donde están las cascadas LBP de OpenCV, si existen.
def get_lbp_dir(haar_dir: Optional[str]) -> Optional[str]:              # intenta encontrar cascadas lbp
    """Best-effort to locate LBP cascades alongside haar cascades."""
    cand = None
    if haar_dir and "haarcascades" in haar_dir: 
        cand = haar_dir.replace("haarcascades", "lbpcascades") #reemplazamos ruta haar a lbp
        if _existing_dir(cand):
            return cand if cand.endswith(os.sep) else cand + os.sep
    # Fallbacks near cv2 package
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        for c in [
            os.path.join(cv2_dir, "data/lbpcascades"),
            os.path.join(cv2_dir, "lbpcascades"),
            "/opt/anaconda3/share/opencv4/lbpcascades",
            "/usr/local/share/opencv4/lbpcascades",         #intentar otras rutas 
        ]:
            if _existing_dir(c):
                return c if c.endswith(os.sep) else c + os.sep
    except Exception:
        pass
    return None

 # [Función que intenta abrir imágenes en varios formatos (JPG, PNG, HEIC, AVIF, etc.).
def imread_any(path: str):   #define una funcion que ayuda a abrir fotos desde un archivo sin importar su formato
    """Robust image loader with AVIF/HEIC support if plugins are installed.
    Order:
      1) cv2.imread. 
      2) cv2.imdecode(raw)
      3) Pillow (incl. AVIF/HEIC via pillow-avif-plugin / pillow-heif if present)
      4) imageio.v3.imread (if available)
    In DEBUG mode, prints file size and magic bytes.
    Returns BGR numpy array or None.
    """
    try:
        # Quick stat + raw read
        try:
            st = os.stat(path)
            size = st.st_size    #metodo para abrir
        except Exception:
            size = -1
        raw = None
        try:
            with open(path, 'rb') as f:
                raw = f.read() 
        except Exception:
            raw = None
        if raw is not None and DEBUG_IMAGE_PATH and path == DEBUG_IMAGE_PATH:
            head = raw[:12] if len(raw) >= 12 else raw           #manjea muchos formatos
            print(f"[DEBUG] file size: {size} bytes | magic: {head}")

        # 1) OpenCV direct 
        img = cv2.imread(path) #abrir en raw
        if img is not None: 
            return img

        # 2) OpenCV from bytes
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            try:
                img2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img2 is not None:
                    return img2
            except Exception:
                pass

        # Enable AVIF/HEIC for Pillow if plugin is present  - si pillow esta instalado abre, sino no  
        try:
            import pillow_avif  # registers AVIF with Pillow - maneja imagenes en formato heic o heif 
        except Exception:
            pass
        try:
            from pillow_heif import register_heif_opener  # HEIC/HEIF/AVIF
            try:
                register_heif_opener()
            except Exception:
                pass
        except Exception:
            pass

        # 3) Pillow
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        except Exception:
            pass
        try:
            pil = Image.open(path)
            if pil.mode not in ("RGB", "RGBA"):
                pil = pil.convert("RGB")
            arr3 = np.array(pil)  # RGB or RGBA
            if arr3.ndim == 2:
                arr3 = np.stack([arr3, arr3, arr3], axis=-1)
            elif arr3.shape[2] == 4:
                arr3 = arr3[:, :, :3]
            return arr3[:, :, ::-1]  # RGB→BGR
        except Exception:
            pass

        # 4) imageio (often supports AVIF if installed)
        try:
            import imageio.v3 as iio
            try:
                arr4 = iio.imread(path)
                if arr4.ndim == 2:
                    arr4 = np.stack([arr4, arr4, arr4], axis=-1)
                elif arr4.shape[2] == 4:
                    arr4 = arr4[:, :, :3]
                return arr4[:, :, ::-1]  # RGB→BGR
            except Exception:
                pass
        except Exception:
            pass

        return None
    except Exception:
        return None


# skin tone utilities (optional; do not affect face decision)
# simple skin tone estimation 
# Quick helper to detect approximate skin fraction and tone (very light to dark)

def get_skin_tone(img_path: str):
    img = imread_any(img_path)
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 40, 60), (50, 255, 255)),
        cv2.inRange(ycc, (0, 85, 135), (255, 135, 180))
    )
    skin_ratio = cv2.countNonZero(mask) / mask.size
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, b = lab[:, :, 0], lab[:, :, 2]
    ita = np.degrees(np.arctan2(np.mean(L[mask > 0]) - 50, np.mean(b[mask > 0]) + 1e-5)) if cv2.countNonZero(mask) else 0
    tone = ("very_light" if ita > 55 else "light" if ita > 41 else "medium" if ita > 10 else "dark")
    return {"skin_frac": round(skin_ratio, 3), "ita": round(float(ita), 2), "tone": tone}

# --- Helper: check if a file exists robustly ---
# Verifica si un archivo existe en la ruta especificada.
def _exists_file(p: str) -> bool:  #verificar si el archivo existe
    try:
        return os.path.isfile(p)
    except Exception:
        return False

ALL_EXTS = ["png","jpg","jpeg","webp","gif","bmp","tif","tiff","jfif",
    "avif","heif","heic"] #reune todas las extensiones q el programa considera


def resolve_avatar_dirs(avatars_dir: str):  
    """Return a list of directories to search for images.
    If avatars_dir contains 'Art_avatars' or 'Fashion_avatars', include them.
    Always include avatars_dir itself if it contains files.
    """
    dirs = []
    try:
        # If avatars_dir itself has files, keep it
        has_files = any(
            os.path.isfile(os.path.join(avatars_dir, f)) for f in os.listdir(avatars_dir) #agrega a la lista de directorios
        )
        if has_files:
            dirs.append(avatars_dir)
    except Exception:
        pass

    # Known subfolders
    for sub in ["Art_avatars", "Fashion_avatars"]:
        p = os.path.join(avatars_dir, sub)
        if os.path.isdir(p):
            dirs.append(p)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            result.append(d)
    # If nothing found, at least return the original
    return result or [avatars_dir]


# --- Fast recursive file index for robust matching ---
def _walk_all_files(avatar_dirs):
    if isinstance(avatar_dirs, str):
        avatar_dirs = [avatar_dirs]
    files = []
    for root_dir in avatar_dirs:
        for root, _, fnames in os.walk(root_dir):
            for fn in fnames:
                files.append(os.path.join(root, fn))
    return files


def build_avatar_index(avatar_dirs):
    """Scan all avatar_dirs recursively and build lookup indexes.
    Returns a dict with:
      - by_name: exact basename (lower) -> full path
      - by_stem: exact stem (lower) -> full path
      - by_id: numeric id string -> full path (if filename starts with that id or stem equals id)
      - all: list of (basename_lower, fullpath)
    """
    if isinstance(avatar_dirs, str):
        avatar_dirs = [avatar_dirs]
    all_paths = _walk_all_files(avatar_dirs)
    by_name, by_stem, by_id, all_list = {}, {}, {}, []
    for p in all_paths:
        base = os.path.basename(p)
        base_l = base.lower()
        stem, _ext = os.path.splitext(base_l)
        all_list.append((base_l, p))
        by_name[base_l] = p
        by_stem[stem] = p
        # If the stem starts with digits, capture as a project_id candidate
        i = 0
        while i < len(stem) and stem[i].isdigit():
            i += 1
        if i > 0:
            pid = stem[:i]
            by_id[pid] = p
        # Also if the entire stem is digits, map that too
        if stem.isdigit():
            by_id[stem] = p
    return {"by_name": by_name, "by_stem": by_stem, "by_id": by_id, "all": all_list}


def _search_dirs_for_patterns(avatar_dirs, patterns):
    if isinstance(avatar_dirs, str):
        avatar_dirs = [avatar_dirs]
    for d in avatar_dirs:
        for pat in patterns:
            hits = glob.glob(os.path.join(d, pat))
            if hits:
                return hits[0]
    return None


def find_image_for_project(avatar_dirs, project_id: int) -> Optional[str]:
    """
    Search across one or many avatar directories for files named by project_id.
    Uses a fast index if available.
    """
    pid = str(project_id)
    # First: index fast path (by_id exact)
    # We expect the caller to pass an index via avatar_dirs if available; accept both for backward compatibility.
    index = None
    if isinstance(avatar_dirs, dict) and "by_id" in avatar_dirs:
        index = avatar_dirs
    if index is not None:
        hit = index["by_id"].get(pid)
        if hit:
            return hit
        # Stricter fallback: exact stem match OR startswith pid followed by a non-digit or delimiter
        for base_l, p in index["all"]:
            stem, _ext = os.path.splitext(base_l)
            if stem == pid:
                return p
        for base_l, p in index["all"]:
            stem, _ext = os.path.splitext(base_l)
            if stem.startswith(pid):
                nxt = stem[len(pid):len(pid)+1]
                if nxt and not nxt.isdigit() and nxt in {"_", "-", ".", " ", "("}:
                    return p
        # No permissive substring matches; return None so we try URL/creator_link instead
        # print(f"[match] No strict match for id/stem; skipping permissive substring for safety.")
        return None
    # Backward fallback (non-indexed): use prior pattern search
    pid = str(project_id)
    patterns = [f"{pid}.{ext}" for ext in ALL_EXTS]
    hit = _search_dirs_for_patterns(avatar_dirs, patterns)
    if hit:
        return hit
    patterns = [f"{pid}.*", f"{pid}_*.*", f"{pid}-*.*"]
    hit = _search_dirs_for_patterns(avatar_dirs, patterns)
    if hit:
        return hit
    if isinstance(avatar_dirs, str):
        avatar_dirs = [avatar_dirs]
    for d in avatar_dirs:
        try:
            for fname in os.listdir(d):
                if pid in fname:
                    return os.path.join(d, fname)
        except Exception:
            pass
    return None


def _debug_candidates(avatar_dirs, project_id: int):
    pid = str(project_id)
    if isinstance(avatar_dirs, str):
        avatar_dirs = [avatar_dirs]
    out = []
    for d in avatar_dirs:
        try:
            files = os.listdir(d)
            hits = [f for f in files if pid in f]
            out.append((d, hits[:5]))
        except Exception:
            out.append((d, ["<error listing>"]))
    return out


def find_image_by_url(avatar_dirs, url: Optional[str]) -> Optional[str]:
    """
    If Step 3 saved avatars using the original filename from the URL, try to match by URL basename.
    Handles querystrings and URL encoding. Uses index if available.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        index = None
        if isinstance(avatar_dirs, dict) and "by_name" in avatar_dirs:
            index = avatar_dirs
        path = urlsplit(url).path
        base = unquote(os.path.basename(path)).lower()
        if not base:
            return None
        if index is not None:
            # Exact basename
            hit = index["by_name"].get(base)
            if hit:
                return hit
            stem, ext = os.path.splitext(base)
            # Prefer exact basename
            for base_l, p in index["all"]:
                if base_l == base:
                    return p
            # Strict startswith: stem followed by common delimiters (to handle versioned filenames)
            for base_l, p in index["all"]:
                if base_l.startswith(stem):
                    nxt = base_l[len(stem):len(stem)+1]
                    if nxt and nxt in {"_", "-", ".", " ", "("}:
                        return p
            # print(f"[match] No strict match for url stem; skipping permissive substring for safety.")
            return None
        # Non-index fallback
        hit = _search_dirs_for_patterns(avatar_dirs, [base])
        if hit:
            return hit
        stem, ext = os.path.splitext(base)
        patterns = []
        if stem:
            patterns.extend([f"{stem}*{ext}", f"{stem}*.*", f"*{stem}*{ext}"])
        hit = _search_dirs_for_patterns(avatar_dirs, patterns)
        return hit
    except Exception:
        return None

# --- Match by creator_link numeric id ---
def find_image_by_creator_link(avatar_dirs, url: Optional[str]) -> Optional[str]:
    """Try to match by numeric creator/user id derived from creator_link URL.
    Example: https://www.kickstarter.com/profile/73488600 -> tries files starting with 73488600
    Uses index if available.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        path = urlsplit(url).path  # e.g., '/profile/73488600' or '/profile/name-73488600'
        # grab the last run of digits anywhere in the path
        import re
        m = re.search(r"(\d+)(?!.*\d)", path)
        if not m:
            return None
        cid = m.group(1)  # '73488600'
        # If we have an index, use it
        index = avatar_dirs if isinstance(avatar_dirs, dict) else None
        if index is not None and "by_id" in index:
            # exact id
            p = index["by_id"].get(cid)
            if p:
                return p
            # strict: stem equals id OR startswith id followed by non-digit delimiter
            for base_l, full in index["all"]:
                stem, _ext = os.path.splitext(base_l)
                if stem == cid:
                    return full
            for base_l, full in index["all"]:
                stem, _ext = os.path.splitext(base_l)
                if stem.startswith(cid):
                    nxt = stem[len(cid):len(cid)+1]
                    if nxt and not nxt.isdigit() and nxt in {"_", "-", ".", " ", "("}:
                        return full
            # print(f"[match] No strict match for creator id/stem; skipping permissive substring for safety.")
            return None
        # Non-index fallback
        patterns = [f"{cid}.*", f"*{cid}*.*"]
        return _search_dirs_for_patterns(avatar_dirs, patterns)
    except Exception:
        return None

 # Detección robusta de rostros en imágenes usando cascadas HAAR y LBP.
 # - Carga múltiples cascadas (frontal, perfil, LBP si están disponibles).
 # - Abre la imagen de forma robusta (cv2 y Pillow).
 # - Escala imágenes pequeñas para mejorar la detección.
 # - Aplica la detección en la imagen original, rotaciones de 90°/270° y espejo horizontal.
 # - Valida candidatos con geometría, confirmación de ojos, proporción de piel (en HSV y YCrCb), nariz/boca y veto de logos.
 # - El veto de logos descarta parches con color plano o muchas líneas rectas (típico de logos).
 # - El parámetro loose_fallback permite una validación más permisiva como último recurso.
 # Devuelve True si se detecta un rostro plausible, False si parece un logo.
def detect_face(
    image_path: str,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (24, 24),
    require_skin: bool = True,
    min_skin_frac: float = 0.12,
) -> bool:
    """
    Stronger face detector with compact validation:
      Order: YuNet (ONNX) → DNN Res10-SSD → Haar/LBP.
      For high-confidence boxes (score ≥ high_conf) only geometry is required; otherwise run eye+logo veto.
    """
    img = imread_any(image_path)
    # initialize annotation bucket for this image
    LAST_FACE_RECTS[image_path] = []
    if img is None:
        return False

    h, w = img.shape[:2]
    if min(h, w) < 120:
        s = max(1.0, 160.0 / max(1, min(h, w)))
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    def _prep(x):
        return cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)) if x.ndim == 3 else cv2.equalizeHist(x)

    #helpers 
    def _is_plausible(x, y, ww, hh, W, H):
        if ww < max(20, 0.010 * min(W, H)) or hh < max(20, 0.010 * min(W, H)): 
            return False   #Más grande que antes: no acepta rectángulos diminutos
        ar = ww / max(1.0, hh)
        if ar < 0.70 or ar > 1.40:
            return False  #relación ancho/alto más cerca de un cuadrado.
        if ww * hh > 0.85 * (W * H):
            return False
        return True

    def _skin_frac(bgr, rect):
        if not require_skin:
            return 1.0
        x, y, ww, hh = rect
        roi = bgr[y:y+hh, x:x+ww] # Toma solo la parte de la imagen donde el detector (YuNet, DNN o Haar) cree que hay una cara
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #Convierte esa porción al espacio de color HSV (Hue, Saturation, Value)
        ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb) #Convierte al espacio YCrCb (luminancia + componentes de color rojo y azul).
        mask_hsv = cv2.inRange(hsv, (0, int(0.25*255), int(0.30*255)), (35, int(0.62*255), 255))
        mask_ycc = cv2.inRange(ycc, (0, 95, 145), (255, 125, 170)) #rangos de color en HSV y YCrCb (mas parecidos a piel)
        mask = cv2.bitwise_or(mask_hsv, mask_ycc) #Esto une los píxeles que cumplan cualquiera de los dos criterios
        return float(cv2.countNonZero(mask)) / float(mask.size) #cuenta cuántos píxeles de piel hay

    def _edge_frac_and_flatness(bgr, rect):
        x, y, ww, hh = rect
        roi = bgr[y:y+hh, x:x+ww]
        if roi.size == 0:
            return 0.0, 0.0, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_frac = float(cv2.countNonZero(edges)) / float(edges.size)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_std = float(np.std(hsv[:, :, 1])); v_std = float(np.std(hsv[:, :, 2]))
        return edge_frac, s_std, v_std

    def _corner_density(bgr, rect):
        x, y, ww, hh = rect
        roi = bgr[y:y+hh, x:x+ww]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        try:
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=3)
            n = 0 if pts is None else len(pts)
        except Exception:
            n = 0
        area = max(1, ww * hh)
        return float(n) / float(area)

    def _accept(bgr, rect, score=None):
        x, y, ww, hh = rect
        H, W = bgr.shape[:2]
        if not _is_plausible(x, y, ww, hh, W, H):
            return False
        skin = _skin_frac(bgr, rect)
        if skin < min_skin_frac:
            return False
        # fast logo veto: many sharp corners but low skin → likely logo/text/graphic
        corner_dens = _corner_density(bgr, rect)
        if corner_dens > 0.06 and skin < 0.35:
            return False
        return True

    # Method 0: OpenCV YuNet (ONNX) if available
    # detector de caras basado en una red neuronal (modelo moderno de OpenCV
    def _load_yunet():
        if not hasattr(cv2, 'FaceDetectorYN_create'):
            return None
        if hasattr(_load_yunet, 'net'):
            return _load_yunet.net
        script_dir = os.path.dirname(__file__)
        names = [
            'face_detection_yunet_2023mar.onnx',
            'yunet.onnx',
        ]
        for n in names:
            for base in [script_dir, os.getcwd(), os.path.expanduser('~'), '/usr/share/opencv4/', '/usr/local/share/opencv4/']:
                p = os.path.join(base, n)
                if os.path.isfile(p):
                    try:
                        net = cv2.FaceDetectorYN_create(p, "", (w, h), score_threshold=0.85, nms_threshold=0.3, top_k=5000)
                        _load_yunet.net = net
                        return net
                    except Exception:
                        pass
        _load_yunet.net = None
        return None
    
    #Llama a net.detect(...) y convierte cada resultado en una caja (x, y, w, h). 
    #Luego valida con _accept(...).

    def _yunet_detect(bgr):
        net = _load_yunet()
        if net is None:
            return False
        H, W = bgr.shape[:2]
        try:
            net.setInputSize((W, H))
            results = net.detect(bgr)
        except Exception:
            return False
        if results is None or len(results) < 2:
            return False
        faces = results[1]  # Nx15: x,y,w,h,score,5 landmarks
        for f in faces:
            x, y, ww, hh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            if _accept(bgr, (x, y, ww, hh), score=None):
                try:
                    LAST_FACE_RECTS[image_path].append((x, y, ww, hh))
                except Exception:
                    pass
                return True
        return False

    # Method 1: OpenCV DNN (Res10-SSD)
    #otro detector de caras con deep learning
    # (más viejo que YuNet, pero muy extendido en OpenCV
    def _load_dnn():  ##busca los archivos del modelo
        if hasattr(_load_dnn, "net"):
            return _load_dnn.net
        script_dir = os.path.dirname(__file__)
        names = [("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"),
                 ("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")]
        for prot, caff in names:
            for base in [script_dir, os.getcwd(), os.path.expanduser("~"), "/usr/share/opencv4/", "/usr/local/share/opencv4/"]:
                p1, p2 = os.path.join(base, prot), os.path.join(base, caff)
                if os.path.isfile(p1) and os.path.isfile(p2):
                    try:
                        _load_dnn.net = cv2.dnn.readNetFromCaffe(p1, p2)
                        return _load_dnn.net
                    except Exception:
                        pass
        _load_dnn.net = None
        return None

    def _dnn_detect(bgr):  #valida tamaño/forma y fracción de piel antes de aceptar
        net = _load_dnn()
        if net is None:   
            return False
        H, W = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        try:
            det = net.forward()
        except Exception:
            return False
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < 0.85: #Si el modelo está poco seguro, no la acepta
                continue
            box = det[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            rect = (max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))
            if _accept(bgr, rect, score=None):
                try:
                    LAST_FACE_RECTS[image_path].append(rect)
                except Exception:
                    pass
                return True
        return False

    # Try YuNet → DNN on original + horizontal flip
    for variant in [img, cv2.flip(img, 1)]:
        if _yunet_detect(variant) or _dnn_detect(variant):
            return True

    # Method 2: HAAR/LBP cascades 
    #detectores clásicos de OpenCV basados en patrones 
    # (no son redes neuronales).
    haar_dir, lbp_dir = get_haar_dir(), get_lbp_dir(get_haar_dir())
    cand_files = [
        (haar_dir, [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt_tree.xml",
            "haarcascade_profileface.xml",
        ]),
        (lbp_dir, ["lbpcascade_frontalface_improved.xml", "lbpcascade_frontalface.xml"]),
    ]
    cascades = []
    for base, names in cand_files:
        if not base:
            continue
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    cascades.append(c)
    if not cascades:
        return False

    def _detect(gray, bgr):
        H, W = gray.shape[:2]
        for c in cascades:
            try:
                faces = c.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
            except Exception:
                faces = []
            for (x, y, ww, hh) in faces:
                if _accept(bgr, (x, y, ww, hh), score=None):
                    try:
                        LAST_FACE_RECTS[image_path].append((x, y, ww, hh))
                    except Exception:
                        pass
                    return True
        return False

    gray0 = _prep(img)
    for variant in [(gray0, img), (cv2.flip(gray0, 1), cv2.flip(img, 1))]:
        if _detect(*variant):
            return True

    return False


 # --- SQLite helpers: retry on 'database is locked' ---
def _commit_with_retry(conn, max_tries: int = 6, base_sleep: float = 0.25):
    for attempt in range(max_tries):
        try:
            conn.commit()
            return True
        except sqlite3.OperationalError as e:
            if 'database is locked' in str(e).lower() and attempt < max_tries - 1:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            raise

def _execute_with_retry(cur, sql, params=(), max_tries: int = 6, base_sleep: float = 0.25):
    for attempt in range(max_tries):
        try:
            cur.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if 'database is locked' in str(e).lower() and attempt < max_tries - 1:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            raise

# Procesa una base de datos SQLite, busca la imagen correspondiente a cada fila (por project_id, URL, etc.),
# aplica la detección de rostro/logo y actualiza/agrega la columna IS_LOGO (0=rostro, 1=logo).
def process_sqlite(
    db_path: str,
    avatars_dir: str,
    table: str,
    overwrite: bool,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
):
    avatar_dirs = resolve_avatar_dirs(avatars_dir)
    avatar_index = build_avatar_index(avatar_dirs)
    print(f"Indexed avatar files: {len(avatar_index['all'])}")
    print(f"HAAR dir: {get_haar_dir()} | LBP dir: {get_lbp_dir(get_haar_dir())}")
    try:
        sample_numeric = [base for base, _ in avatar_index['all'] if base.split('.')[0].isdigit()][:5]
        print(f"Index sample numeric stems: {sample_numeric}")
    except Exception:
        pass
    try:
        counts = []
        for d in avatar_dirs:
            try:
                files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
                counts.append((d, len(files), files[:3]))
            except Exception as e:
                counts.append((d, f"error: {e}", []))
        print("avatars search paths:")
        for d, n, samples in counts:
            print(f"  - {d} (files: {n}; samples: {samples})")
    except Exception as e:
        print(f"Warning: could not inspect avatar dirs: {e}")

    # Start fresh: clear marked faces output folder for this run
    _clear_dir_contents(ANNOTATE_OUTPUT_DIR)
    print(f"[MARK] Cleared output folder: {os.path.realpath(ANNOTATE_OUTPUT_DIR)}")
    # --- Diagnostics counters ---
    matched_count = 0
    unreadable_count = 0
    face_count_live = 0

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    cur = conn.cursor()
    # --- Diagnostics: confirm which file is open and writable ---
    abs_db = os.path.realpath(db_path)
    print(f"[DB] Opened: {abs_db}")
    try:
        cur.execute("PRAGMA database_list")
        dblist = cur.fetchall()
        print(f"[DB] database_list: {dblist}")
    except Exception as e:
        print(f"[DB] database_list failed: {e}")

    # Ensure IS_LOGO column exists
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if "IS_LOGO" not in cols:
        _execute_with_retry(cur, f"ALTER TABLE {table} ADD COLUMN IS_LOGO INTEGER")
        _commit_with_retry(conn)
    # --- Print table schema once for diagnostics ---
    try:
        cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
        row = cur.fetchone()
        print(f"[DB] Table schema for {table}: {row[0] if row else '<not found>'}")
    except Exception as e:
        print(f"[DB] Could not read schema for {table}: {e}")

    # Select project_id, creator_avatar, creator_link; if not overwriting, only rows with NULL
    if overwrite:
        cur.execute(f"SELECT project_id, creator_avatar, creator_link FROM {table}")
    else:
        cur.execute(f"SELECT project_id, creator_avatar, creator_link FROM {table} WHERE IS_LOGO IS NULL")

    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} rows to process.")
    try:
        sample_ids = [r[0] if isinstance(r, (list, tuple)) else r for r in rows[:3]]
        print(f"Sample project_ids: {sample_ids}")
    except Exception:
        pass

    debug_left = 5  # print extra diagnostics for first 5 missing matches

    for i, (project_id, creator_avatar, creator_link) in enumerate(rows, start=1):
        matched_via = None
        img_path = find_image_for_project(avatar_index, project_id)
        if img_path:
            matched_via = "project_id"
        if not img_path:
            img_path = find_image_by_url(avatar_index, creator_avatar)
            if img_path:
                matched_via = "creator_avatar URL"
        if not img_path:
            img_path = find_image_by_creator_link(avatar_index, creator_link)
            if img_path:
                matched_via = "creator_link ID"
        if not img_path:
            if debug_left > 0:
                cand = _debug_candidates(avatar_dirs, project_id)
                print(f"  debug: candidates containing '{project_id}': {cand}")
                debug_left -= 1
            # treat missing/unreadable as no detected face → logo
            is_logo = 1
            _execute_with_retry(
                cur,
                f"UPDATE {table} SET IS_LOGO = ? WHERE project_id = ?",
                (is_logo, project_id),
            )
            try:
                if cur.rowcount == 0:
                    print(f"[WARN] No rows updated for project_id={project_id} (image NOT FOUND). Check data types and table name.")
            except Exception:
                pass
            if i % 50 == 0 or i == total:
                _commit_with_retry(conn)
            print(
                f"[{i}/{total}] project_id={project_id} -> image NOT FOUND. Tried: project_id, creator_avatar basename, creator_link id. Set IS_LOGO=1"
            )
            continue
        else:
            matched_count += 1

        # Check readability explicitly
        _arr = imread_any(img_path)
        if _arr is None:
            unreadable_count += 1
            is_logo = 1
            _execute_with_retry(
                cur,
                f"UPDATE {table} SET IS_LOGO = ? WHERE project_id = ?",
                (is_logo, project_id),
            )
            try:
                if cur.rowcount == 0:
                    print(f"[WARN] No rows updated for project_id={project_id} (unreadable image). Check data types and table name.")
            except Exception:
                pass
            if i % 50 == 0 or i == total:
                _commit_with_retry(conn)
            print(f"[{i}/{total}] project_id={project_id} ({matched_via}) -> UNREADABLE image -> IS_LOGO=1 | img={img_path}")
            continue

        has_face = detect_face(
            img_path,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=(min_size, min_size),
            require_skin=True,
            min_skin_frac=0.22,  # exige que haya más porcentaje de píxeles “piel”
        )
        is_logo = 0 if has_face else 1
        _execute_with_retry(
            cur,
            f"UPDATE {table} SET IS_LOGO = ? WHERE project_id = ?",
            (is_logo, project_id),
        )
        try:
            if cur.rowcount == 0:
                print(f"[WARN] No rows updated for project_id={project_id} (normal detection). Check data types and table name.")
        except Exception:
            pass
        if has_face:
            face_count_live += 1
        if has_face:
            try:
                rects = LAST_FACE_RECTS.get(img_path, [])
                save_annotated_face(img_path, rects, ANNOTATE_OUTPUT_DIR, prefix=str(project_id))
            except Exception:
                pass
        if i % 50 == 0 or i == total:
            _commit_with_retry(conn)
        print(
            f"[{i}/{total}] project_id={project_id} ({matched_via}) -> {'FACE' if has_face else 'NO FACE'} -> IS_LOGO={is_logo} | img={img_path}"
        )

    # End-of-run summary
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE IS_LOGO = 0")
        c_face = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE IS_LOGO = 1")
        c_logo = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE IS_LOGO IS NULL")
        c_null = cur.fetchone()[0]
        print(f"Summary → FACE (0): {c_face} | LOGO (1): {c_logo} | NULL: {c_null}")
        print(f"Diag → matched files: {matched_count} | unreadable: {unreadable_count} | faces detected live: {face_count_live}")
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE IS_LOGO IN (0,1)")
            c_done = cur.fetchone()[0]
            print(f"[DB] Rows with IS_LOGO set: {c_done}")
        except Exception:
            pass
    except Exception as e:
        print(f"Summary unavailable: {e}")

    try:
        dbg = os.environ.get("DEBUG_PID")
        if dbg:
            print(f"DEBUG_PID was set to {dbg}. To inspect a single image, set DEBUG_IMAGE_PATH at the top and rerun _debug_single_image(DEBUG_IMAGE_PATH).")
    except Exception:
        pass
    _commit_with_retry(conn)
    try:
        print(f"[DB] total_changes in this run: {conn.total_changes}")
    except Exception:
        pass
    conn.close()
    print("SQLite update complete.")


 # [ES]
 # Procesa un archivo Excel, busca la imagen correspondiente a cada fila (por project_id, URL, etc.),
 # aplica la detección de rostro/logo y actualiza/agrega la columna IS_LOGO (0=rostro, 1=logo).
def process_excel(
    xlsx_path: str,
    avatars_dir: str,
    sheet: str,
    overwrite: bool,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
):
    # Load worksheet
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    avatar_dirs = resolve_avatar_dirs(avatars_dir)
    avatar_index = build_avatar_index(avatar_dirs)
    print(f"Indexed avatar files: {len(avatar_index['all'])}")
    print(f"HAAR dir: {get_haar_dir()} | LBP dir: {get_lbp_dir(get_haar_dir())}")
    try:
        counts = []
        for d in avatar_dirs:
            try:
                files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
                counts.append((d, len(files), files[:3]))
            except Exception as e:
                counts.append((d, f"error: {e}", []))
        print("avatars search paths:")
        for d, n, samples in counts:
            print(f"  - {d} (files: {n}; samples: {samples})")
    except Exception as e:
        print(f"Warning: could not inspect avatar dirs: {e}")

    # Start fresh: clear marked faces output folder for this run
    _clear_dir_contents(ANNOTATE_OUTPUT_DIR)
    print(f"[MARK] Cleared output folder: {os.path.realpath(ANNOTATE_OUTPUT_DIR)}")

    if "project_id" not in df.columns:
        raise ValueError("The Excel sheet must include a 'project_id' column.")

    if "IS_LOGO" not in df.columns:
        df["IS_LOGO"] = pd.Series([None] * len(df), dtype="float")

    has_url_col = 'creator_avatar' in df.columns

    # Determine which rows to process
    if overwrite:
        idxs = df.index.tolist()
    else:
        idxs = df.index[df["IS_LOGO"].isna()].tolist()

    total = len(idxs)
    print(f"Found {total} rows to process.")
    try:
        sample_ids = [df.at[idx, "project_id"] for idx in idxs[:3]]
        print(f"Sample project_ids: {sample_ids}")
    except Exception:
        pass

    debug_left = 5  # print extra diagnostics for first 5 missing matches

    for j, idx in enumerate(idxs, start=1):
        project_id = int(df.at[idx, "project_id"])  # ensure numeric
        img_path = find_image_for_project(avatar_index, project_id)
        matched_via = "project_id"
        if not img_path and has_url_col:
            img_path = find_image_by_url(avatar_index, df.at[idx, 'creator_avatar'])
            if img_path:
                matched_via = "creator_avatar URL"
        if not img_path:
            if debug_left > 0:
                cand = _debug_candidates(avatar_dirs, project_id)
                print(f"  debug: candidates containing '{project_id}': {cand}")
                debug_left -= 1
            df.at[idx, "IS_LOGO"] = 1  # treat missing/unreadable as no detected face → logo
            print(
                f"[{j}/{total}] project_id={project_id} -> image NOT FOUND. Searched indexed dirs. Set IS_LOGO=1"
            )
            continue

        has_face = detect_face(
            img_path,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=(min_size, min_size),
            require_skin=True,
            min_skin_frac=0.22,
        )
        df.at[idx, "IS_LOGO"] = 0 if has_face else 1
        if has_face:
            try:
                rects = LAST_FACE_RECTS.get(img_path, [])
                save_annotated_face(img_path, rects, ANNOTATE_OUTPUT_DIR, prefix=str(project_id))
            except Exception:
                pass
        print(
            f"[{j}/{total}] project_id={project_id} ({matched_via}) -> {'FACE' if has_face else 'NO FACE'} -> IS_LOGO={df.at[idx, 'IS_LOGO']}"
        )

    # Save back (overwrites file)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)

    print("Excel update complete.")


 # [ES] Punto de entrada principal; recibe argumentos y decide si trabajar con SQLite o Excel.
def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Face vs. Logo (Haar Cascade, CPU)."
    )
    parser.add_argument("--data", required=True, help="Path to dataset (.db or .xlsx).")
    parser.add_argument("--avatars_dir", required=True, help="Folder with downloaded avatars.")
    parser.add_argument("--table", default="Links", help="SQLite table name (default: Links).")
    parser.add_argument("--sheet", default="Sheet1", help="Excel sheet name (default: Sheet1).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite IS_LOGO for all rows.",
    )
    parser.add_argument(
        "--scale_factor", type=float, default=1.1, help="Haar scaleFactor (default: 1.1)."
    )
    parser.add_argument(
        "--min_neighbors", type=int, default=4, help="Haar minNeighbors (default: 4)."
    )
    parser.add_argument(
        "--min_size", type=int, default=24, help="Minimum face size in px (default: 24)."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.avatars_dir):
        raise FileNotFoundError(f"avatars_dir not found: {args.avatars_dir}")

    ext = os.path.splitext(args.data)[1].lower()
    if ext == ".db":
        process_sqlite(
            db_path=args.data,
            avatars_dir=args.avatars_dir,
            table=args.table,
            overwrite=args.overwrite,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=args.min_size,
        )
    elif ext in (".xlsx", ".xls"):
        process_excel(
            xlsx_path=args.data,
            avatars_dir=args.avatars_dir,
            sheet=args.sheet,
            overwrite=args.overwrite,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=args.min_size,
        )
    else:
        raise ValueError("Unsupported data file type. Use .db or .xlsx.")


# [ES] Ejecución directa (sin usar CLI); aquí defines las rutas para pruebas.
if __name__ == "__main__":
    RUN_DIRECT = True  # change to False to use CLI arguments instead
    if RUN_DIRECT:
        print("OpenCV:", cv2.__version__)

        # --- Single-image debug: set DEBUG_IMAGE_PATH above to test detection quickly ---
        if DEBUG_IMAGE_PATH and os.path.isfile(DEBUG_IMAGE_PATH):
            _debug_single_image(DEBUG_IMAGE_PATH)
            sys.exit(0)

        print("[RUN] Using database:", os.path.realpath("/Users/plandi/Downloads/DB/Art_Links_V2.db"))
        process_sqlite(
            db_path="/Users/plandi/Downloads/DB/Art_Links_V2.db",
            avatars_dir="/Users/plandi/Downloads/DB/Avatars",
            table="Links",
            overwrite=True,
            scale_factor=1.14,
            min_neighbors=7,
            min_size=36,
        )
    else:
        main()
