import pandas as pd
import numpy as np
import sqlite3

import concurrent.futures as futures
import logging
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse, unquote, urljoin

import requests
import bs4
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from PIL import Image


art_db_path = "/Users/plandi/Downloads/DB/Art_Links.db"  #same folder
fashion_db_path = "/Users/plandi/Downloads/DB/Fashion_Links.db"
art_conn = sqlite3.connect(art_db_path) # Connect to the SQLite database
fashion_conn = sqlite3.connect(fashion_db_path)

art_df = pd.read_sql_query("Select * from LINKS", art_conn)
fashion_df = pd.read_sql_query( "Select * from LINKS", fashion_conn)


# ----------------- User / path configuration -----------------
# Replace these with your actual dataframes if they are not named this way
# art_df, fashion_df = <already loaded>

BASE_DIR = Path.cwd()
LOG_FILE = BASE_DIR / "download.log"
ART_DIR = BASE_DIR / "art_avatars_full"
FASHION_DIR = BASE_DIR / "fashion_avatars_full"

# ----------------- Logging (console + file) -----------------
logger = logging.getLogger("avatar_downloader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# clear previous handlers if re-running in notebook
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
ch.setLevel(logging.WARNING)  # keep INFO in file, show WARNING+ in console so tqdm bar isn't spammed

# ----------------- Helpers -----------------
def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.4,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS"])
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.kickstarter.com/",
    })
    return session

_IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|gif|webp|bmp|svg)(?:[?#].*)?$", re.I)

def looks_like_image_url(url: str) -> bool:
    try:
        path = urlparse(url).path or ""
        return bool(_IMAGE_EXT_RE.search(path))
    except Exception:
        return False



def init_driver(timeout=10):
    """Return a headless undetected_chromedriver Chrome instance or None on failure."""
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1200,800")
        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(timeout)
        return driver
    except Exception as e:
        logger.warning(f"Selenium driver init failed: {e}. Will skip Selenium page parsing.")
        return None



def get_extension_from_response(response: requests.Response, url: str = "") -> str:
    """Return normalized extension (including dot) using Content-Type or URL path fallback."""
    ct = response.headers.get("Content-Type", "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "gif" in ct:
        return ".gif"
    if "webp" in ct:
        return ".webp"
    if "svg" in ct:
        return ".svg"
    if "bmp" in ct:
        return ".bmp"

    # fallback: try to parse extension from URL
    try:
        path = unquote(urlparse(url).path or "")
        ext = Path(path).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            return ".jpg"
        if ext in [".png", ".gif", ".webp", ".svg", ".bmp"]:
            return ext
    except Exception:
        pass

    # ultimate fallback
    return ".jpg"

from urllib.parse import parse_qsl, urlencode, urlunparse

def upgrade_avatar_url(url: str, target_size: int = 160) -> str:
    """
    IMPORTANT:
    - If the URL is *signed* (has `sig=`), do NOT change width/height — the signature is tied to them.
      Changing them causes 403. Just return the original URL.
    - If the URL is *unsigned*, strip any width/height params to let the CDN return a higher-res default.
      (We avoid forcing a size to stay within allowed patterns.)
    """
    if not isinstance(url, str):
        return url
    try:
        parsed = urlparse(url)
        q_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        has_sig = any(k.lower() == "sig" for k, _ in q_pairs)
        if has_sig:
            # Don't touch signed URLs; altering dims invalidates the signature -> 403
            return url
        # Unsigned: remove width/height entirely
        new_q = [(k, v) for k, v in q_pairs if k.lower() not in {"width", "height"}]
        new_query = urlencode(new_q, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        return url


def get_highres_from_creator_page(creator_link: str) -> str | None:
    """
    Requests + BeautifulSoup only (no Selenium).
    Looks for og:image, twitter:image, JSON-LD image, then <img> with avatar/profile hints.
    Prefers URLs that already encode >=160px in width/height (when present).
    """
    if not isinstance(creator_link, str) or not creator_link.startswith("http"):
        return None

    session = make_session()
    try:
        r = session.get(creator_link, timeout=(4, 12), allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logger.debug(f"creator page fetch failed {creator_link}: {e}")
        return None

    soup = bs4.BeautifulSoup(html, "html.parser")

    def _norm(u: str) -> str | None:
        if not u:
            return None
        u = u.strip()
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            u = urljoin(creator_link, u)
        return u if u.startswith("http") else None

    candidates = []

    # og:image
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        u = _norm(og["content"])
        if u:
            candidates.append(u)

    # twitter:image (+ :src variants)
    for name in [("name", "twitter:image"), ("property", "twitter:image"),
                 ("name", "twitter:image:src"), ("property", "twitter:image:src")]:
        tag = soup.find("meta", attrs={name[0]: name[1]})
        if tag and tag.get("content"):
            u = _norm(tag["content"])
            if u:
                candidates.append(u)

    # JSON-LD blocks (look for 'image')
    import json
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "{}")
        except Exception:
            continue
        def _collect(d):
            if isinstance(d, dict):
                if "image" in d:
                    img = d["image"]
                    if isinstance(img, str):
                        u = _norm(img);  u and candidates.append(u)
                    elif isinstance(img, list):
                        for it in img:
                            u = _norm(str(it)); u and candidates.append(u)
                    elif isinstance(img, dict) and "url" in img:
                        u = _norm(img.get("url")); u and candidates.append(u)
                for v in d.values():
                    _collect(v)
            elif isinstance(d, list):
                for v in d:
                    _collect(v)
        _collect(data)

    # <img> fallback (prefer 'avatar'/'profile' hints; largest from srcset)
    for img in soup.find_all("img"):
        cls = " ".join(img.get("class", [])) if img.get("class") else ""
        alt = (img.get("alt") or "").lower()
        if any(k in cls.lower() for k in ["avatar", "profile", "user", "creator"]) or \
           any(k in alt for k in ["avatar", "profile", "creator"]):
            srcset = img.get("srcset")
            if srcset:
                parts = [p.strip() for p in srcset.split(",") if p.strip()]
                if parts:
                    candidates.append(parts[-1].split()[0])  # largest in srcset
            src = img.get("src") or img.get("data-src")
            if src:
                candidates.append(src)

    normed = []
    for c in candidates:
        u = _norm(c)
        if u:
            normed.append(u)
    if not normed:
        return None

    # Prefer candidates that explicitly carry a >=160 width/height in query
    from urllib.parse import parse_qs
    def score(u: str) -> tuple[int, int]:
        q = parse_qs(urlparse(u).query)
        w = int(q.get("width", ["0"])[0] or 0)
        h = int(q.get("height", ["0"])[0] or 0)
        return (max(w, h), min(w, h))
    normed.sort(key=score, reverse=True)

    best = normed[0]
    # Do NOT modify signed URLs; if unsigned, strip small dims
    return upgrade_avatar_url(best, target_size=160)


# ----------------- Extraction (Selenium + BeautifulSoup) -----------------
def extract_image_urls(rows: List[Dict]) -> List[Tuple[int, str, str]]:
    """
    Given rows = [{'project_id':..., 'creator_avatar':...}, ...],
    return list of (project_id, final_image_url, creator_link).
    Uses Selenium+BS only for rows where the link isn't obviously an image.
    """
    results: List[Tuple[int, str, str]] = []
    # Try to initialize Selenium once (used only for non-image links)
    driver = None  # keep Selenium off to avoid anti-bot triggering and flicker
    session = make_session()  # used to HEAD-check some URLs cheaply if needed

    try:
        iterator = rows
        for r in tqdm(iterator, desc="Extracting image URLs", unit="row", dynamic_ncols=True):
            pid = r["project_id"]
            url = r["creator_avatar"]
            creator_link = r.get("creator_link") or ""
            if not isinstance(url, str) or not url.lower().startswith("http"):
                logger.warning(f"Skipping invalid URL for project_id {pid}: {url}")
                continue

            # If URL looks like an image already, use it directly.
            if looks_like_image_url(url):
                upgraded = upgrade_avatar_url(url, target_size=160)
                results.append((pid, upgraded, creator_link))
                continue

            # Try a quick HEAD with session — sometimes original link redirects to an image
            try:
                head = session.head(url, timeout=6, allow_redirects=True)
                ct = head.headers.get("Content-Type", "").lower()
                if ct.startswith("image/"):
                    upgraded = upgrade_avatar_url(url, target_size=160)
                    results.append((pid, upgraded, creator_link))
                    continue
            except Exception:
                # fall back to Selenium parsing
                pass

            # If we have a driver, load the page and try to find an <img>
            img_url = None
            if driver:
                try:
                    driver.get(url)
                    time.sleep(0.25)
                    soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
                    # heuristics: look for img in page, prefer images with srcset or profile/avatar classes
                    img_candidates = soup.find_all("img")
                    if img_candidates:
                        # pick first reasonably sized one or with 'avatar'/'profile' in class/alt
                        chosen = None
                        for img in img_candidates:
                            src = img.get("src") or img.get("data-src")
                            if not src:
                                continue
                            cls = " ".join(img.get("class", [])) if img.get("class") else ""
                            alt = (img.get("alt") or "").lower()
                            if "avatar" in cls.lower() or "avatar" in alt or "profile" in cls.lower() or "profile" in alt:
                                chosen = src
                                break
                        if not chosen:
                            chosen = img_candidates[0].get("src") or img_candidates[0].get("data-src")
                        img_url = urljoin(url, chosen)
                except TimeoutException:
                    logger.warning(f"Selenium timeout when loading {url} (project {pid})")
                except WebDriverException as e:
                    logger.warning(f"Selenium/WebDriver error for {url} (project {pid}): {e}")
                except Exception as e:
                    logger.debug(f"Selenium parsing exception for {url}: {e}")

            # final fallback: just use original url if nothing else found
            if not img_url:
                logger.info(f"Could not parse an <img> for project {pid}, falling back to original URL.")
                img_url = url

            img_url = upgrade_avatar_url(img_url, target_size=160)
            results.append((pid, img_url, creator_link))
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
        session.close()

    return results



# ----------------- Downloader -----------------
def download_images(pairs: List[Tuple[int, str, str]], save_dir: Path, max_workers: int = 8):
    save_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    def _download_one(item: Tuple[int, str, str]) -> bool:
        pid, primary_url, creator_link = item

        # If a file exists but is too small, we'll try to replace it
        existing = list(save_dir.glob(f"{pid}.*"))
        if existing:
            keep = True
            for fp in existing:
                try:
                    with Image.open(fp) as im:
                        w, h = im.size
                    if min(w, h) < 160:
                        keep = False
                        try:
                            fp.unlink()
                        except Exception:
                            pass
                except Exception:
                    keep = False
                    try:
                        fp.unlink()
                    except Exception:
                        pass
            if keep:
                logger.info(f"Skipping existing: {pid}")
                return True

        attempts = []
        # If we have a creator page, try to discover a bigger signed URL first
        if creator_link:
            hi = get_highres_from_creator_page(creator_link)
            if hi:
                attempts.append(hi)
        attempts.append(primary_url)

        last_fp = None
        for url_try in attempts:
            try:
                resp = session.get(url_try, timeout=15, stream=True, allow_redirects=True)
                if resp.status_code == 403:
                    continue  # try next candidate
                resp.raise_for_status()

                ext = get_extension_from_response(resp, url_try)
                filename = f"{pid}{ext}"
                filepath = save_dir / filename

                with open(filepath, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                last_fp = filepath

                # Validate size; if still small, try next attempt
                try:
                    with Image.open(filepath) as im:
                        w, h = im.size
                    if min(w, h) < 160:
                        continue  # try next candidate before upscaling
                    else:
                        logger.info(f"Downloaded {filename} ({w}x{h})")
                        return True
                except Exception:
                    # if we cannot open, try next attempt
                    continue
            except Exception:
                continue

        # If we downloaded something but it's still small or unreadable, upscale as last resort
        if last_fp and last_fp.exists():
            try:
                with Image.open(last_fp) as im:
                    w, h = im.size
                    if min(w, h) < 160:
                        if w <= h:
                            new_w = 160
                            new_h = int(h * (160 / w))
                        else:
                            new_h = 160
                            new_w = int(w * (160 / h))
                        up = im.resize((new_w, new_h), Image.LANCZOS)
                        up.save(last_fp)
                        logger.info(f"Downloaded {last_fp.name} and upscaled to {new_w}x{new_h}")
                        return True
                    else:
                        logger.info(f"Downloaded {last_fp.name} ({w}x{h})")
                        return True
            except Exception:
                pass

        logger.error(f"Failed for project_id {pid}: all attempts exhausted.")
        return False

    results = []
    total = len(pairs)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_download_one, p) for p in pairs]
        for f in tqdm(futures.as_completed(futs), total=total, desc=f"Downloading to {save_dir.name}", dynamic_ncols=True):
            results.append(f.result())

    session.close()
    success = sum(1 for r in results if r)
    logger.info(f"Downloaded {success} / {total} files to {save_dir}")
    return success, total




# ----------------- Main runner (for a dataframe) -----------------
def process_df(df: pd.DataFrame, folder: Path):
    # Build unique rowsß
    rows = df[["project_id", "creator_avatar", "creator_link"]].drop_duplicates().to_dict("records")
    # Extract image URLs (Selenium + BS when needed)
    pairs = extract_image_urls(rows)   # list of (project_id, image_url, creator_link)
    # Download concurrently
    download_images(pairs, folder, max_workers=8)

# ----------------- Example run -----------------
# Make sure art_df and fashion_df exist in the kernel.
# If they are named differently, replace below references.

if "art_df" in globals():
    logger.info("Starting Art avatars download")
    process_df(art_df, ART_DIR)
else:
    logger.warning("art_df not found in globals — skipping Art download")

if "fashion_df" in globals():
    logger.info("Starting Fashion avatars download")
    process_df(fashion_df, FASHION_DIR)
else:
    logger.warning("fashion_df not found in globals — skipping Fashion download")

logger.info("All done. See download.log for details.")
