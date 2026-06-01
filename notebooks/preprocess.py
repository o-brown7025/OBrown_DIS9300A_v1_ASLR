import os
import json
import shutil
import subprocess
import cv2
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple

RAW_DIR = "raw_videos_mp4"
OUT_DIR = "videos"
JSON_PATH = "WLASL_v0.3.json"

MAX_ATTEMPTS = 2
HARD_TIMEOUT_SECONDS = 45          # per clip extraction/copy
FFMPEG_TIMEOUT_SECONDS = 90        # per conversion file

DEFAULT_FPS = 25
FOURCC = "mp4v"
COPY_FULL_IF_ENDFRAME_LEQ_ZERO = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_probably_video(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 1024:
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


# ----------------------------
# Robust conversion (replaces swf2mp4.sh)
# ----------------------------
def ffmpeg_available() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], text=True)
        return bool(out)
    except Exception:
        return False


def convert_to_mp4_in_place(raw_dir: str) -> None:
    """
    Convert .swf/.mkv/.webm -> .mp4 using ffmpeg, with hard timeouts and skip-on-failure.
    Outputs mp4 next to the source file. Does NOT delete the source by default.
    """
    ensure_dir(raw_dir)

    if not ffmpeg_available():
        print("[WARN] ffmpeg not found; skipping conversion step.")
        return

    exts = {".swf", ".mkv", ".webm"}
    candidates = []
    for fn in os.listdir(raw_dir):
        _, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            candidates.append(os.path.join(raw_dir, fn))

    if not candidates:
        print("[INFO] No .swf/.mkv/.webm files found to convert.")
        return

    for src in candidates:
        base, _ = os.path.splitext(src)
        dst = base + ".mp4"
        if os.path.exists(dst) and os.path.getsize(dst) > 1024 and is_probably_video(dst):
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i", src,
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            dst
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=FFMPEG_TIMEOUT_SECONDS,
                check=True
            )
            if not is_probably_video(dst):
                try:
                    os.remove(dst)
                except Exception:
                    pass
                print(f"[SKIP] Conversion produced invalid mp4: {dst}")
            else:
                print(f"[OK] Converted -> {dst}")
        except subprocess.TimeoutExpired:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except Exception:
                pass
            print(f"[SKIP] ffmpeg timeout converting: {src}")
        except Exception:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except Exception:
                pass
            print(f"[SKIP] ffmpeg failed converting: {src}")


# ----------------------------
# Clip extraction (fast + hard timeout)
# ----------------------------
def safe_copy(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)


def write_frames_to_video(frames_iter, dst_path: str, size: Tuple[int, int], fps: int = DEFAULT_FPS) -> None:
    ensure_dir(os.path.dirname(dst_path))
    out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*FOURCC), fps, size)
    wrote_any = False
    try:
        for frame in frames_iter:
            if frame is None:
                continue
            out.write(frame)
            wrote_any = True
    finally:
        out.release()

    if not wrote_any:
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except Exception:
            pass
        raise RuntimeError("No frames written.")


def iter_frames_range(src_video_path: str, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Failed to open source video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    idx = start_frame
    try:
        while idx <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
            idx += 1
    finally:
        cap.release()


def process_instance(src_video_path: str, dst_video_path: str, start_frame: int, end_frame: int) -> None:
    if not is_probably_video(src_video_path):
        raise RuntimeError("Source missing or not readable video.")

    if COPY_FULL_IF_ENDFRAME_LEQ_ZERO and end_frame <= 0:
        safe_copy(src_video_path, dst_video_path)
        return

    if start_frame < 0:
        start_frame = 0
    if end_frame < start_frame:
        raise RuntimeError("Invalid frame range.")

    it = iter_frames_range(src_video_path, start_frame, end_frame)
    first = next(it, None)
    if first is None:
        raise RuntimeError("No frames in requested range.")

    h, w = first.shape[:2]
    size = (w, h)

    def frames():
        yield first
        for f in it:
            yield f

    write_frames_to_video(frames(), dst_video_path, size=size, fps=DEFAULT_FPS)


@dataclass
class TaskResult:
    ok: bool
    error: Optional[str] = None


def _task_worker(q: mp.Queue, fn, args):
    try:
        fn(*args)
        q.put(TaskResult(ok=True))
    except Exception as e:
        q.put(TaskResult(ok=False, error=str(e)))


def run_with_hard_timeout(fn, args, timeout_seconds: int) -> TaskResult:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_task_worker, args=(q, fn, args))
    p.daemon = True
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join(2)
        return TaskResult(ok=False, error=f"Hard timeout after {timeout_seconds}s")

    if q.empty():
        return TaskResult(ok=False, error="Worker exited without result")
    return q.get()


def extract_all_instances(content) -> None:
    ensure_dir(OUT_DIR)

    cnt = 0
    for entry in content:
        for inst in entry.get("instances", []):
            cnt += 1
            url = (inst.get("url") or "").strip()
            video_id = str(inst.get("video_id") or "").strip()
            if not video_id:
                continue

            is_yt = ("youtube" in url) or ("youtu.be" in url)
            if is_yt:
                yt_identifier = url[-11:] if len(url) >= 11 else ""
                if not yt_identifier:
                    continue
                src = os.path.join(RAW_DIR, yt_identifier + ".mp4")
            else:
                src = os.path.join(RAW_DIR, video_id + ".mp4")

            dst = os.path.join(OUT_DIR, video_id + ".mp4")

            if os.path.exists(dst):
                continue
            if not os.path.exists(src):
                continue

            start_frame = int(inst.get("frame_start", 1)) - 1
            end_frame = int(inst.get("frame_end", 0)) - 1

            success = False
            last_err = None

            for _ in range(MAX_ATTEMPTS):
                if not is_probably_video(src):
                    last_err = "Dead/non-video source"
                    break

                result = run_with_hard_timeout(
                    process_instance,
                    args=(src, dst, start_frame, end_frame),
                    timeout_seconds=HARD_TIMEOUT_SECONDS
                )

                if result.ok and os.path.exists(dst) and os.path.getsize(dst) > 1024:
                    success = True
                    break

                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                except Exception:
                    pass

                last_err = result.error or "Unknown failure"

            if success:
                print(f"[{cnt}] OK -> {dst}")
            else:
                print(f"[{cnt}] SKIP -> {video_id} ({last_err})")


def main() -> None:
    ensure_dir(RAW_DIR)

    # Convert legacy formats that exist in RAW_DIR to mp4
    convert_to_mp4_in_place(RAW_DIR)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        content = json.load(f)

    extract_all_instances(content)


if __name__ == "__main__":
    main()
