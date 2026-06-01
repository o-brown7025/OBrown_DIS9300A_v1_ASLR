import os
import json
import sys
import time
import logging
import urllib.request
import urllib.error
import subprocess
import shutil
from typing import Optional
import cv2  # for quick post-download validation

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    filename=f"download_{int(time.time())}.log",
    filemode="w",
    level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# ----------------------------
# Config
# ----------------------------
INDEXFILE = "WLASL_v0.3.json"
SAVE_DIR = "raw_videos_mp4"          # aligns with preprocess.py
YOUTUBE_DOWNLOADER = os.getenv("YOUTUBE_DOWNLOADER", "yt-dlp")

MAX_ATTEMPTS = 2

HTTP_TIMEOUT_SECONDS = 20            # urllib timeout (non-yt)
YTDLP_TIMEOUT_SECONDS = 60           # hard timeout (yt-dlp subprocess)

MIN_BYTES = 1024

NON_VIDEO_EXTS = {
    ".html", ".htm", ".php", ".jsp", ".asp", ".aspx",
    ".txt", ".json", ".xml",
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp3", ".wav"
}

ACCEPTED_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".swf"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def url_extension(url: str) -> str:
    u = url.split("?", 1)[0].split("#", 1)[0]
    _, ext = os.path.splitext(u)
    return ext.lower()


def looks_like_dead_or_nonvideo(url: str) -> bool:
    ext = url_extension(url)
    if ext in NON_VIDEO_EXTS:
        return True
    if ext and ext not in ACCEPTED_VIDEO_EXTS:
        return True
    return False


def is_probably_video_file(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < MIN_BYTES:
            return False
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False


def request_bytes(url: str, referer: str = "", timeout: int = HTTP_TIMEOUT_SECONDS) -> bytes:
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    headers = {"User-Agent": user_agent}
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, None, headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        # skip obvious HTML/text responses
        if ctype and ("text/html" in ctype or "text/plain" in ctype):
            raise RuntimeError(f"Non-video content-type: {ctype}")
        return resp.read()


def save_bytes(data: bytes, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def select_method(url: str) -> str:
    u = (url or "").lower()
    if "aslpro" in u:
        return "aslpro"
    if "youtube" in u or "youtu.be" in u:
        return "youtube"
    return "direct"


# ----------------------------
# Non-YouTube downloaders
# ----------------------------
def download_aslpro(url: str, dirname: str, video_id: str) -> None:
    dst = os.path.join(dirname, f"{video_id}.swf")
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
        logging.info(f"[ASLPRO] exists -> {dst}")
        return

    data = request_bytes(
        url,
        referer="http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi",
        timeout=HTTP_TIMEOUT_SECONDS
    )
    save_bytes(data, dst)

    if os.path.getsize(dst) < MIN_BYTES:
        try:
            os.remove(dst)
        except Exception:
            pass
        raise RuntimeError("Downloaded SWF too small / likely invalid.")


def download_direct_video(url: str, dirname: str, video_id: str) -> None:
    ext = url_extension(url) or ".mp4"

    if looks_like_dead_or_nonvideo(url):
        raise RuntimeError(f"Skipping non-video URL by extension: {ext}")

    dst = os.path.join(dirname, f"{video_id}{ext}")
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
        logging.info(f"[DIRECT] exists -> {dst}")
        return

    data = request_bytes(url, timeout=HTTP_TIMEOUT_SECONDS)
    save_bytes(data, dst)

    # Validate only for common video containers
    if ext in {".mp4", ".mkv", ".webm"}:
        if not is_probably_video_file(dst):
            try:
                os.remove(dst)
            except Exception:
                pass
            raise RuntimeError("Downloaded file is not a readable video (possibly HTML error).")


# ----------------------------
# YouTube support (graceful)
# ----------------------------
def youtube_downloader_available() -> bool:
    """
    Returns True if YOUTUBE_DOWNLOADER is runnable.
    Does NOT raise; we want graceful degradation.
    """
    try:
        # shutil.which handles PATH resolution
        if shutil.which(YOUTUBE_DOWNLOADER) is None:
            return False
        out = subprocess.check_output([YOUTUBE_DOWNLOADER, "--version"], text=True).strip()
        return bool(out)
    except Exception:
        return False


def download_youtube(url: str, dirname: str) -> None:
    """
    yt-dlp runner with:
    - hard timeout
    - no retries
    - socket timeout
    - no partials
    """
    ensure_dir(dirname)

    yt_id = url[-11:] if len(url) >= 11 else ""
    if yt_id:
        for ext in (".mp4", ".mkv", ".webm"):
            existing = os.path.join(dirname, yt_id + ext)
            if os.path.exists(existing) and os.path.getsize(existing) >= MIN_BYTES:
                logging.info(f"[YT] exists -> {existing}")
                return

    cmd = [
        YOUTUBE_DOWNLOADER,
        url,
        "-o", os.path.join(dirname, "%(id)s.%(ext)s"),
        "--socket-timeout", "15",
        "--retries", "0",
        "--fragment-retries", "0",
        "--retry-sleep", "0",
        "--no-continue",
        "--no-part",
        "--no-warnings",
        "--no-progress",
        "--quiet",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=YTDLP_TIMEOUT_SECONDS
    )

    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed rc={proc.returncode}: {proc.stderr[-500:]}")

    if yt_id:
        candidates = [os.path.join(dirname, yt_id + e) for e in (".mp4", ".mkv", ".webm")]
        candidates = [p for p in candidates if os.path.exists(p)]
        if not candidates:
            raise RuntimeError("yt-dlp reported success but no output file was found.")

        candidates.sort(key=lambda p: (0 if p.endswith(".mp4") else 1))
        best = candidates[0]

        if not is_probably_video_file(best):
            try:
                os.remove(best)
            except Exception:
                pass
            raise RuntimeError("Downloaded YouTube output is not a readable video.")


# ----------------------------
# Orchestrator
# ----------------------------
def download_one(gloss: str, url: str, video_id: str, out_dir: str, yt_enabled: bool) -> None:
    url = (url or "").strip()
    video_id = str(video_id or "").strip()
    if not url or not video_id:
        return

    method = select_method(url)

    # Skip obvious dead URLs early (except YouTube)
    if method != "youtube" and looks_like_dead_or_nonvideo(url):
        logging.info(f"[SKIP] non-video/dead by extension -> {video_id} url={url}")
        return

    if method == "youtube" and not yt_enabled:
        logging.warning(
            f"[SKIP-YT] yt-dlp not available. Skipping YouTube video_id={video_id} url={url}"
        )
        return

    last_err: Optional[str] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logging.info(f"[{attempt}/{MAX_ATTEMPTS}] gloss={gloss} video_id={video_id} url={url}")

            if method == "aslpro":
                download_aslpro(url, out_dir, video_id)
            elif method == "youtube":
                download_youtube(url, out_dir)
            else:
                download_direct_video(url, out_dir, video_id)

            return  # success

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = f"Network error: {e}"
        except subprocess.TimeoutExpired:
            last_err = f"Hard timeout: yt-dlp exceeded {YTDLP_TIMEOUT_SECONDS}s"
        except Exception as e:
            last_err = str(e)

        # No slow retry; also don't waste attempt #2 on known non-video content
        if last_err and ("Non-video content-type" in last_err or "Skipping non-video URL" in last_err):
            break

    logging.error(f"[FAIL] gloss={gloss} video_id={video_id} reason={last_err}")


def download_all(indexfile: str, out_dir: str) -> None:
    ensure_dir(out_dir)

    with open(indexfile, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Decide if we even need yt-dlp
    needs_youtube = any(
        ("youtube" in (inst.get("url") or "").lower() or "youtu.be" in (inst.get("url") or "").lower())
        for entry in content
        for inst in entry.get("instances", [])
    )

    yt_enabled = False
    if needs_youtube:
        yt_enabled = youtube_downloader_available()
        if not yt_enabled:
            logging.warning(
                f"yt-dlp not found. Non-YouTube videos will download; YouTube will be skipped.\n"
                f"To enable YouTube downloads in Colab:\n"
                f"  !pip -q install yt-dlp\n"
                f"or\n"
                f"  !apt-get -y update && apt-get -y install yt-dlp\n"
            )

    total = 0
    for entry in content:
        gloss = entry.get("gloss", "")
        for inst in entry.get("instances", []):
            total += 1
            try:
                download_one(gloss, inst.get("url", ""), inst.get("video_id", ""), out_dir, yt_enabled)
            except Exception as e:
                logging.error(f"[SKIP] unexpected error video_id={inst.get('video_id','')}: {e}")

    logging.info(f"Done. Processed instances: {total}")


if __name__ == "__main__":
    logging.info("Start downloading videos (YouTube + non-YouTube).")
    download_all(INDEXFILE, SAVE_DIR)
