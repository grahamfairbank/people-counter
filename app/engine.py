"""Capture -> detect -> track -> count pipeline.

Two threads: one drains the stream and keeps only the newest frame (so a slow
CPU never builds up lag), the other runs YOLO + ByteTrack on that frame at the
target rate. A track becomes a counted person once it has been seen
`min_hits` times; a colour-histogram descriptor merges someone who leaves and
comes back within the re-entry window into the same person, and so does
reappearing close to where someone vanished a few seconds earlier (colour alone
fails for distant people and for front-versus-back views).
"""
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import Config, mask_url
from .store import THUMB_DIR, Store

MODEL = os.environ.get("MODEL", "yolo11n.pt")
IMGSZ = int(os.environ.get("IMGSZ", "640"))
TRACK_TTL = 2.0  # seconds a track may go undetected before it is closed
PREVIEW_WIDTH = 960
MAX_SAMPLES = 12  # appearance samples kept per pass through the frame
QUICK_RETURN_DIST = 0.25  # of the frame: how near the exit point a quick returner must reappear

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
torch.set_num_threads(int(os.environ.get("THREADS", "3")))


def describe(crop: np.ndarray) -> np.ndarray | None:
    """Clothing-colour descriptor: hue/saturation and brightness histograms over
    three horizontal stripes of the centre of the box. Each block sums to 1."""
    h, w = crop.shape[:2]
    if h < 30 or w < 12:
        return None
    hsv = cv2.cvtColor(crop[:, int(w * 0.2) : max(int(w * 0.8), int(w * 0.2) + 1)], cv2.COLOR_BGR2HSV)
    blocks = []
    for stripe in np.array_split(hsv, 3, axis=0):
        hs = cv2.calcHist([stripe], [0, 1], None, [12, 6], [0, 180, 0, 256]).ravel()
        v = cv2.calcHist([stripe], [2], None, [8], [0, 256]).ravel()
        blocks += [hs / max(hs.sum(), 1), v / max(v.sum(), 1)]
    return np.concatenate(blocks).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection averaged over the six blocks: 0 (disjoint) .. 1 (same)."""
    return float(np.minimum(a, b).sum() / 6)


class Track:
    def __init__(self, ts: float, pos: tuple[float, float]):
        self.first = self.last = ts
        self.entry = self.exit = pos  # feet position, as fractions of the frame
        self.near = None  # person who vanished close by moments before this track began
        self.hits = 0
        self.samples = []  # descriptors spread over this pass
        self.every = 1  # sample every Nth hit; doubles as the track gets long
        self.prior = []  # samples from earlier visits once merged
        self.match = None  # best-matching recently departed person
        self.match_sim = 0.0
        self.fresh = False  # counted as a new person by this very track
        self.thumb = None
        self.thumb_score = 0.0
        self.thumb_dirty = False
        self.pid = None
        self.synced = ts


class Engine:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.started = time.time()
        self.state = "starting"
        self.error = ""
        self.resolution = None
        self.capture_fps = 0.0
        self.infer_ms = 0.0
        self.process_fps = 0.0
        self.last_frame_ts = 0.0
        self.in_frame = 0
        self.jpeg = None
        self.jpeg_seq = 0

        self._running = True
        self._frame = None
        self._frame_seq = 0
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stream_gen = 0
        self._tracks: dict[int, Track] = {}
        # Recently departed people: appearance samples, when and where last seen.
        self._gallery: dict[int, tuple[list[np.ndarray], float, tuple[float, float]]] = {}
        self._model = None
        self._model_version = -1
        self._last_prune = 0.0

        threading.Thread(target=self._capture_loop, daemon=True, name="capture").start()
        threading.Thread(target=self._process_loop, daemon=True, name="process").start()

    def stop(self):
        self._running = False

    # ---- capture -----------------------------------------------------------

    def _capture_loop(self):
        backoff = 2
        while self._running:
            source = self.cfg.source
            if not source:
                self.state, self.error = "unconfigured", ""
                time.sleep(1)
                continue
            self.state = "connecting"
            cap = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000],
            )
            if not cap.isOpened():
                cap.release()
                self.state = "error"
                self.error = f"Could not open {mask_url(source)}"
                self._sleep_unless_source_changes(source, backoff)
                backoff = min(backoff * 2, 30)
                continue

            backoff = 2
            self._stream_gen += 1
            self.state, self.error = "streaming", ""
            # Local files (used for testing) decode faster than real time; pace them.
            is_file = "://" not in source
            delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if is_file else 0
            stamps = deque(maxlen=50)
            while self._running and self.cfg.source == source:
                ok, frame = cap.read()
                if not ok:
                    self.error = "Stream ended or stalled; reconnecting"
                    break
                now = time.time()
                stamps.append(now)
                if len(stamps) > 1:
                    self.capture_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0] + 1e-6)
                self.resolution = (frame.shape[1], frame.shape[0])
                with self._frame_lock:
                    self._frame = (frame, now, self._stream_gen)
                    self._frame_seq += 1
                self._frame_event.set()
                if delay:
                    time.sleep(delay)
            cap.release()
            self.capture_fps = 0.0
            if self._running and self.cfg.source == source:
                self.state = "error"
                time.sleep(2)

    def _sleep_unless_source_changes(self, source: str, seconds: float):
        deadline = time.time() + seconds
        while self._running and time.time() < deadline and self.cfg.source == source:
            time.sleep(0.5)

    # ---- detection + tracking ---------------------------------------------

    def _load_model(self):
        """(Re)build the model so tracker thresholds follow the current settings."""
        cfg = self.cfg
        self._model_version = cfg.version
        tracker = Path("/tmp/tracker.yaml")
        tracker.write_text(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {cfg.conf}\n"
            "track_low_thresh: 0.1\n"
            f"new_track_thresh: {cfg.conf}\n"
            f"track_buffer: {max(int(TRACK_TTL * cfg.fps), 2)}\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
        )
        self._tracker = str(tracker)
        self._model = YOLO(MODEL)
        self._close_all_tracks()

    def _process_loop(self):
        last_seq, last_run, gen = 0, 0.0, -1
        while self._running:
            self._frame_event.wait(0.5)
            self._frame_event.clear()
            now = time.time()
            if now - self._last_prune > 3600:
                self._last_prune = now
                self.store.prune_thumbs()
            with self._frame_lock:
                item, seq = self._frame, self._frame_seq
            if item is None or seq == last_seq:
                self._expire(now)
                continue
            wait = 1 / self.cfg.fps - (now - last_run)
            if wait > 0:
                time.sleep(wait)
                with self._frame_lock:
                    item, seq = self._frame, self._frame_seq
            frame, ts, frame_gen = item
            last_seq = seq
            try:
                if self._model is None or self._model_version != self.cfg.version:
                    self._load_model()
                elif frame_gen != gen:
                    # New connection: old track ids no longer mean anything.
                    self._close_all_tracks()
                    for t in getattr(self._model.predictor, "trackers", []):
                        t.reset()
                gen = frame_gen
                started = time.time()
                self._process(frame, ts)
                if last_run:
                    self.process_fps = 0.8 * self.process_fps + 0.2 / max(started - last_run, 1e-3)
                last_run = started
                self.last_frame_ts = ts
            except Exception as e:  # keep the loop alive; surface the problem in the UI
                self.error = f"Processing error: {e}"
                time.sleep(1)

    def _process(self, frame: np.ndarray, ts: float):
        cfg = self.cfg
        t0 = time.perf_counter()
        boxes = self._model.track(
            frame, persist=True, classes=[0], conf=0.1, imgsz=IMGSZ, tracker=self._tracker, verbose=False
        )[0].boxes
        self.infer_ms = 0.8 * self.infer_ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.infer_ms else (
            time.perf_counter() - t0
        ) * 1000

        h, w = frame.shape[:2]
        drawn = []
        if boxes.id is not None:
            for xyxy, tid, conf in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
                x1, y1 = max(int(xyxy[0]), 0), max(int(xyxy[1]), 0)
                x2, y2 = min(int(xyxy[2]), w), min(int(xyxy[3]), h)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                tr = self._tracks.get(tid)
                pos = ((x1 + x2) / 2 / w, y2 / h)
                if tr is None:
                    tr = self._tracks[tid] = Track(ts, pos)
                    self._find_near(tr)
                tr.last, tr.exit = ts, pos
                tr.hits += 1
                crop = frame[y1:y2, x1:x2]
                desc = describe(crop)
                if desc is not None and tr.hits % tr.every == 0:
                    tr.samples.append(desc)
                    if len(tr.samples) > MAX_SAMPLES:
                        tr.samples, tr.every = tr.samples[::2], tr.every * 2
                    if tr.pid is None or tr.fresh:
                        self._match(tr, desc, ts)
                score = conf * (x2 - x1) * (y2 - y1)
                if score > tr.thumb_score * 1.2:
                    tr.thumb, tr.thumb_score, tr.thumb_dirty = crop.copy(), score, True
                if tr.pid is None and tr.hits >= cfg.min_hits:
                    self._resolve(tr)
                elif tr.fresh and (pid := self._returning(tr)) is not None:
                    # Looked new at first, but a later view matches someone who just left.
                    self._merge(tr, pid)
                elif tr.pid is not None and ts - tr.synced > 5:
                    self._sync(tr)
                drawn.append((x1, y1, x2, y2, tr))
        self.in_frame = len(drawn)
        self._expire(ts)
        self._render(frame, drawn)

    def _match(self, tr: Track, desc: np.ndarray, now: float):
        """Remember the recently departed person this track resembles most."""
        window = self.cfg.reentry_minutes * 60
        for pid, (samples, last, _) in list(self._gallery.items()):
            if now - last > window:
                del self._gallery[pid]
                continue
            sim = max((similarity(desc, s) for s in samples), default=0.0)
            if sim > tr.match_sim:
                tr.match, tr.match_sim = pid, sim

    def _find_near(self, tr: Track):
        """Did someone vanish close to where this track just appeared, moments ago?"""
        best = QUICK_RETURN_DIST
        for pid, (_, last, pos) in self._gallery.items():
            if tr.first - last <= self.cfg.quick_seconds:
                dist = math.hypot(tr.entry[0] - pos[0], tr.entry[1] - pos[1])
                if dist < best:
                    tr.near, best = pid, dist

    def _returning(self, tr: Track) -> int | None:
        """The departed person this track should be merged into, if any."""
        if tr.near in self._gallery:
            return tr.near
        if tr.match is not None and tr.match not in self._gallery:
            tr.match, tr.match_sim = None, 0.0  # expired, or claimed by another track
        if tr.match is not None and tr.match_sim >= self.cfg.similarity:
            return tr.match
        return None

    def _resolve(self, tr: Track):
        """A track just qualified: is it a returning person or a new one?"""
        if (pid := self._returning(tr)) is not None:
            self._merge(tr, pid)
        else:
            tr.pid = self.store.add(tr.first, tr.last)
            tr.fresh = True
            self._sync(tr)

    def _merge(self, tr: Track, into: int):
        tr.prior = self._gallery.pop(into)[0]
        if tr.pid is not None:
            self.store.delete(tr.pid)
            (THUMB_DIR / f"{tr.pid}.jpg").unlink(missing_ok=True)
        tr.pid, tr.fresh = into, False
        tr.thumb_dirty = False  # keep the first visit's thumbnail
        self.store.revisit(tr.pid, tr.last, tr.match_sim if into == tr.match else None)
        tr.synced = tr.last

    def _sync(self, tr: Track):
        tr.synced = tr.last
        self.store.touch(tr.pid, tr.last)
        if tr.thumb_dirty and tr.thumb is not None:
            tr.thumb_dirty = False
            th = tr.thumb
            scale = 200 / th.shape[0]
            if scale < 1:
                th = cv2.resize(th, (max(int(th.shape[1] * scale), 1), 200), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(THUMB_DIR / f"{tr.pid}.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _close(self, tid: int):
        tr = self._tracks.pop(tid)
        if tr.pid is not None:
            self._sync(tr)
            self.store.log_pass(
                tr.pid, tr.first, tr.last, tr.hits, tr.match_sim or None, not tr.fresh,
                np.stack(tr.samples).tobytes() if tr.samples else None,
                tr.entry, tr.exit, tr.pid == tr.near,
            )
            self._gallery[tr.pid] = ((tr.prior + tr.samples)[-2 * MAX_SAMPLES :], tr.last, tr.exit)

    def _expire(self, now: float):
        for tid in [t for t, tr in self._tracks.items() if now - tr.last > TRACK_TTL]:
            self._close(tid)
        if not self._tracks:
            self.in_frame = 0

    def _close_all_tracks(self):
        for tid in list(self._tracks):
            self._close(tid)

    def _render(self, frame: np.ndarray, drawn: list):
        scale = min(PREVIEW_WIDTH / frame.shape[1], 1)
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        for x1, y1, x2, y2, tr in drawn:
            counted = tr.pid is not None
            colour = (214, 120, 42) if counted else (160, 160, 160)
            p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(img, p1, p2, colour, 2)
            label = f"#{tr.pid}" if counted else "..."
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 8), (p1[0] + tw + 8, p1[1]), colour, -1)
            cv2.putText(img, label, (p1[0] + 4, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
            self.jpeg_seq += 1

    # ---- status ------------------------------------------------------------

    def status(self) -> dict:
        now = time.time()
        tracks = list(self._tracks.values())
        return {
            "state": self.state,
            "error": self.error,
            "source": mask_url(self.cfg.source),
            "resolution": self.resolution,
            "capture_fps": round(self.capture_fps, 1),
            "process_fps": round(self.process_fps, 1),
            "infer_ms": round(self.infer_ms),
            "frame_age": round(now - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "in_frame": self.in_frame,
            "pending": sum(1 for t in tracks if t.pid is None),
            "remembered": len(self._gallery),
            "uptime": round(now - self.started),
            "model": MODEL,
            "now": now,
        }
