import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import Config
from .engine import Engine
from .store import THUMB_DIR, Store

STATIC = Path(__file__).parent / "static"
MAX_BUCKETS = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = Config()
    app.state.store = Store()
    app.state.engine = Engine(app.state.cfg, app.state.store)
    yield
    app.state.engine.stop()


app = FastAPI(title="People Counter", lifespan=lifespan)


class Settings(BaseModel):
    source: str | None = None
    fps: float | None = None
    conf: float | None = None
    min_hits: int | None = None
    reentry_minutes: float | None = None
    similarity: float | None = None
    quick_seconds: float | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    store = app.state.store
    now = time.time()
    return {
        **app.state.engine.status(),
        "last_hour": store.total_since(now - 3600),
        "last_24h": store.total_since(now - 86400),
        "all_time": store.total_since(0),
    }


@app.get("/api/counts")
def counts(start: float, end: float, bucket: float):
    """Unique people per bucket. The browser picks start/bucket so buckets align to its local time."""
    if bucket <= 0 or end <= start or (end - start) / bucket > MAX_BUCKETS:
        raise HTTPException(400, "Invalid range")
    found = app.state.store.counts(start, end, bucket)
    n = int(-(-(end - start) // bucket))
    return {"start": start, "bucket": bucket, "counts": [found.get(i, 0) for i in range(n)]}


@app.get("/api/people")
def people(limit: int = 30):
    return app.state.store.recent(max(1, min(limit, 200)))


@app.get("/api/config")
def get_config():
    return app.state.cfg.as_dict()


@app.post("/api/config")
def set_config(settings: Settings):
    app.state.cfg.update(settings.model_dump(exclude_none=True))
    return app.state.cfg.as_dict()


@app.get("/thumbs/{pid}.jpg")
def thumb(pid: int):
    path = THUMB_DIR / f"{pid}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=30"})


@app.get("/snapshot.jpg")
def snapshot():
    jpeg = app.state.engine.jpeg
    if jpeg is None:
        raise HTTPException(503, "No frame yet")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/stream.mjpg")
async def stream():
    engine = app.state.engine

    async def frames():
        seq = -1
        while True:
            if engine.jpeg is not None and engine.jpeg_seq != seq:
                seq = engine.jpeg_seq
                jpeg = engine.jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
