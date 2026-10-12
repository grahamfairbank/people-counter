# People Counter

Counts unique people walking through one camera's RTSP feed. CPU only.

- Web UI on port 8095: live view, discovery status, counts per timeframe, settings
- Set the camera URL in the UI's Settings panel, or as `RTSP_URL` in `.env`
- Data lives in `./data`: `people.db` (SQLite), `thumbs/`, `config.json` (settings saved from the UI)

```bash
cp .env.example .env
docker compose up -d --build
```

Then open `http://<host>:8095`. The UI has no login; keep it on a trusted network.
A low-resolution sub stream is usually the right source: frames are scaled to
640 pixels for detection anyway, and decoding a 4K stream costs far more CPU.

## How it counts

YOLO11n finds people in ~5 frames per second and ByteTrack follows each one across
the frame. Someone seen in `Sightings before counting` analysed frames becomes a
counted person. When a person leaves, clothing-colour samples from their pass are
kept for the re-entry window; a new arrival who matches is treated as the same
person (a second visit) rather than counted again. So is anyone who appears close
to where a person vanished within the quick-return window, which covers distant
people and front-versus-back views where colour fails. Both are heuristics, not
face or body recognition: two people in similar clothes can be merged, and one
person seen from a very different angle can be counted twice.

## Re-tuning later

`passes` in `people.db` has one row per walk through the frame, including the raw
appearance samples, the best match score seen and where it entered and left the
frame, so the re-entry settings can be
re-evaluated offline against real traffic. `persons` has one row per unique person.
