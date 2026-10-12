# People Counter

Counts the unique people who walk through one camera's RTSP feed, and shows the
result on a small web dashboard. It runs in a single Docker container on an
ordinary CPU; no GPU is needed.

- **Unique people, not detections.** Each person is followed across the frame and
  counted once. Someone who steps out and comes back shortly after is recognised
  as a return visit instead of a new person.
- **Counts by timeframe.** Last hour, last 24 hours, today, last 7 or 30 days, or
  any custom range, as a chart or a table.
- **Live status.** An annotated live view plus stream state, frame rates,
  detection time and errors, so you can see what the detector is doing.
- **Self-contained.** One container, a SQLite file and a folder of thumbnails.
  Nothing leaves your network.

## Contents

- [Quick start](#quick-start)
- [Choosing a camera stream](#choosing-a-camera-stream)
- [The dashboard](#the-dashboard)
- [How counting works](#how-counting-works)
- [Settings](#settings)
- [Performance](#performance)
- [Accuracy and limits](#accuracy-and-limits)
- [Data and privacy](#data-and-privacy)
- [Re-tuning from recorded data](#re-tuning-from-recorded-data)
- [HTTP API](#http-api)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Licence](#licence)

## Quick start

You need Docker with the Compose plugin, a camera that serves RTSP, and about
3 GB of disk for the image.

```bash
git clone https://github.com/grahamfairbank/people-counter.git
cd people-counter
cp .env.example .env
docker compose up -d --build
```

The first build downloads PyTorch (CPU build) and the detector weights and takes
several minutes. Then open `http://<host>:8095`.

The page starts in the "No camera configured" state with the Settings panel
open. Paste the camera's RTSP URL, save, and the status should change to
"Streaming" within a few seconds. You can instead set `RTSP_URL` in `.env`
before starting.

To update later:

```bash
git pull
docker compose up -d --build
```

## Choosing a camera stream

Most IP cameras offer a high-resolution main stream and a low-resolution sub
stream. **Use the sub stream.** Every frame is scaled to 640 pixels wide before
detection, so a 4K stream adds decoding cost without adding detail the detector
can use. See [Performance](#performance) for the measured difference.

The exception is a wide scene where people are far away: on a 640×360 stream a
distant person can be only 30 pixels tall, which is at the edge of what can be
detected.

URL formats differ by manufacturer. For Reolink cameras, for example:

```text
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_sub     sub stream
rtsp://USER:PASSWORD@CAMERA_IP:554/h265Preview_01_main    main stream
rtsp://USER:PASSWORD@CAMERA_IP:554/Preview_01_sub         newer firmware
```

Two common stumbling blocks:

- RTSP is switched off by default on some cameras and has to be enabled in the
  camera's network settings.
- Characters such as `@`, `#`, `:` and `/` in the password must be URL-encoded
  (`@` becomes `%40`).

## The dashboard

| Section | What it shows |
|---|---|
| Status pill | Streaming, Connecting, Stream error, or No camera configured |
| Live view | The feed with a box around each person: blue with a person number once counted, grey while still unconfirmed |
| Discovery status | Source (password hidden), resolution, camera and analysis frame rates, detection time, people in frame, unconfirmed tracks, people being watched for re-entry, uptime, and the last error |
| Tiles | People in frame now; unique people in the last hour, last 24 hours, and all time |
| Unique people by timeframe | Preset and custom ranges as a bar chart, with a table view. Buckets are 5 minutes, 1 hour or 1 day depending on the range, in the browser's local time |
| Most recent people | A thumbnail of each counted person, when they were first seen, how long they were in view, and how many visits were merged into them |
| Settings | Camera URL and the tuning values described under [Settings](#settings) |

## How counting works

```mermaid
flowchart LR
    A[RTSP stream] --> B[Keep newest frame]
    B --> C[Detect people<br/>YOLO11n]
    C --> D[Follow each person<br/>ByteTrack]
    D --> E{Seen in enough<br/>frames?}
    E -- no --> D
    E -- yes --> F{Matches someone<br/>who just left?}
    F -- yes --> G[Add a visit to<br/>that person]
    F -- no --> H[Count a<br/>new person]
```

1. **Capture.** One thread reads the stream continuously and keeps only the
   newest frame, so a slow CPU drops frames instead of falling behind.
2. **Detect.** A second thread runs the YOLO11n detector on the newest frame at
   the analysis rate (5 frames per second by default), looking only for people.
3. **Track.** ByteTrack links detections from frame to frame into tracks. A
   track survives up to 2 seconds without a detection, which covers brief
   occlusions.
4. **Confirm.** A track becomes a counted person once it has been detected in
   enough analysed frames (5 by default, about one second). This filters out
   one-frame false detections.
5. **Merge returns.** When a person leaves, the counter remembers them for the
   re-entry window. A new track is merged into a remembered person, and recorded
   as another visit, if either test passes:
   - **Position.** The track appears within the quick-return window (30 seconds)
     and within a quarter of the frame of where that person vanished. This
     handles track breaks, distant people, and someone who walks out of view and
     straight back.
   - **Appearance.** Clothing-colour samples from the track match samples from
     the remembered person above the strictness threshold. Samples are colour
     histograms of three horizontal bands of the person, taken throughout each
     pass, and the best pairing between the two sets is used.

   If a person was already counted as new and a later view matches someone
   remembered, the new record is removed and folded into the earlier one.

A person's count belongs to the time they were **first** seen. Someone who
arrives at 9:58 and is still around at 10:05 counts in the 9:00 hour only.

## Settings

All of these can be changed in the dashboard's Settings panel and take effect
immediately. The environment variables set the starting values.

| Setting in the UI | Environment variable | Default | Meaning |
|---|---|---|---|
| Camera stream URL | `RTSP_URL` | empty | The stream to watch |
| Analysis rate | `TARGET_FPS` | `5` | Frames analysed per second (1–15). Higher follows fast movers better and uses more CPU |
| Detection confidence | `CONFIDENCE` | `0.45` | Minimum detector confidence to start following someone. Raise it if objects are mistaken for people |
| Sightings before counting | `MIN_HITS` | `5` | Analysed frames a person must appear in before being counted |
| Re-entry window (minutes) | `REENTRY_MINUTES` | `5` | How long a departed person is remembered for appearance matching. `0` disables it |
| Re-entry match strictness | `SIMILARITY` | `0.7` | Clothing-colour similarity (0–1) needed to merge. Lower merges more readily |
| Quick-return window (seconds) | `QUICK_RETURN_SECONDS` | `30` | How soon someone must reappear near a vanishing point to be merged by position. `0` disables it |

> **Saved settings win.** The first time you save from the UI, all values are
> written to `data/config.json`, and from then on that file overrides the
> environment variables. Delete it to go back to the values in `.env`.

These are set by environment variable only:

| Environment variable | Default | Meaning |
|---|---|---|
| `THREADS` | `3` | CPU threads used for detection. Raise it if you give the machine more cores |
| `MODEL` | `yolo11n.pt` | Ultralytics model. Larger models (`yolo11s.pt`) are more accurate and several times slower; they are downloaded on first start |
| `IMGSZ` | `640` | Detector input size. Larger sees smaller people and costs more CPU |
| `DATA_DIR` | `/data` | Where the database, thumbnails and saved settings live inside the container |

The web port is 8095; change the left-hand side of `ports` in
`docker-compose.yml` to publish it elsewhere.

## Performance

Measured on a 4-core virtual machine (Intel Xeon D-1541, 2.1 GHz) with
`THREADS=3`, analysing 5 frames per second:

| Stream | Container CPU | Detection time per frame | Memory |
|---|---|---|---|
| 640×360 at 10 fps (sub stream) | about 1.2 cores | 73–95 ms | about 580 MiB |
| 3840×2160 at 25 fps (main stream) | about 2.6 cores | about 132 ms | about 640 MiB |

Most of the difference is video decoding: the 4K stream is decoded in full even
though only one frame in five is analysed.

To use less CPU, lower the analysis rate; the cost falls roughly in proportion.
The Docker image is about 2.5 GB.

## Accuracy and limits

This is a practical counter, not an identity system. It does not recognise
faces or bodies, and its numbers are estimates.

- **Similar clothes can be merged.** Two different people in similar colours
  inside the re-entry window may be counted as one. Raise the match strictness
  if you see this.
- **One person can be counted twice.** A person who returns after the
  quick-return window, looking different from before (front instead of back,
  sun instead of shade), may be counted again. Lower the strictness, or
  lengthen the windows, if you see this.
- **Crowds undercount.** In dense scenes tracks break often and the merge tests
  become unreliable. The defaults suit low-traffic scenes such as a driveway,
  path or doorway.
- **Distant people are marginal.** People under about 30 pixels tall are
  detected inconsistently and are too small for colour matching, so only the
  position test can merge them.
- **Night and infrared.** Colour matching is weak in monochrome night footage.
- **Restarts forget who just left.** The re-entry memory is held in RAM, so a
  person present across a restart can be counted twice.
- **Daily buckets ignore daylight-saving changes.** They are fixed 24-hour
  spans from local midnight at the start of the range.

The recent-people thumbnails are the quickest way to judge accuracy: duplicates
and wrong merges are visible at a glance.

## Data and privacy

Everything is stored in `./data` on the host:

| Path | Contents |
|---|---|
| `people.db` | SQLite database of counted people and their passes |
| `thumbs/<id>.jpg` | One thumbnail per counted person, deleted after 30 days |
| `config.json` | Settings saved from the UI, **including the camera URL and its password** |

- The dashboard has **no login**. Anyone who can reach port 8095 can see the
  live view and the thumbnails and change the settings. Keep it on a trusted
  network or put it behind a reverse proxy with authentication.
- The camera password is hidden in the dashboard and API responses, but is
  stored in plain text in `config.json` (and `.env`, if set there).
- `data/` and `.env` are git-ignored.
- Recording images of people may be regulated where you live, particularly if
  the camera covers public space. Check what applies to you.

To start over, stop the container and delete `data/`.

### Database

`persons` — one row per unique person:

| Column | Meaning |
|---|---|
| `id` | Person number shown in the UI. Gaps appear where records were merged |
| `first_seen`, `last_seen` | Unix timestamps in seconds |
| `visits` | Passes merged into this person |
| `merge_sim` | Colour similarity of the most recent merge; empty if merged by position or never merged |

`passes` — one row per walk through the frame:

| Column | Meaning |
|---|---|
| `person_id` | The person this pass was assigned to |
| `first_seen`, `last_seen`, `hits` | When it happened and how many analysed frames it was detected in |
| `best_match_sim` | Best colour similarity to any remembered person, whether or not it led to a merge |
| `merged` | 1 if the pass was merged into an existing person |
| `by_position` | 1 if the merge was made by the position test rather than by colour |
| `entry_x`, `entry_y`, `exit_x`, `exit_y` | Where the person's feet entered and left, as fractions of the frame (0–1) |
| `samples` | Colour samples as raw float32, 240 values per sample, up to 12 per pass |

Unique people per day, in the server's time zone:

```bash
sqlite3 data/people.db "SELECT date(first_seen, 'unixepoch', 'localtime') AS day, COUNT(*) FROM persons GROUP BY day"
```

## Re-tuning from recorded data

Because each pass keeps its colour samples and positions, you can test
different settings against real traffic without recording video. This prints
the colour similarity between every pair of passes, the same measure the
counter uses:

```python
import sqlite3
import numpy as np

db = sqlite3.connect("data/people.db")
passes = {
    pid: np.frombuffer(blob, np.float32).reshape(-1, 240)
    for pid, blob in db.execute("SELECT id, samples FROM passes WHERE samples IS NOT NULL")
}

def similarity(a, b):
    return max(float(np.minimum(x, y).sum() / 6) for x in a for y in b)

for i in passes:
    for j in passes:
        if i < j:
            print(i, j, round(similarity(passes[i], passes[j]), 2))
```

Compare the scores for pairs you know are the same person against pairs you
know are different, and set the match strictness between the two groups.

## HTTP API

The dashboard is built on these endpoints; none require authentication.

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Stream state, frame rates, detection time, people in frame, and totals for the last hour, 24 hours and all time |
| `GET /api/counts?start=&end=&bucket=` | Unique people per bucket. `start` and `end` are Unix seconds, `bucket` is the bucket length in seconds; at most 1,000 buckets |
| `GET /api/people?limit=` | Most recent people, newest first (up to 200) |
| `GET /api/config` | Current settings, with the camera password hidden |
| `POST /api/config` | Update any subset of settings, as JSON |
| `GET /thumbs/<id>.jpg` | A person's thumbnail |
| `GET /snapshot.jpg` | The latest annotated frame |
| `GET /stream.mjpg` | The annotated live view as an MJPEG stream |

For example, unique people in the last hour as a single number:

```bash
NOW=$(date +%s); curl -s "http://localhost:8095/api/counts?start=$((NOW-3600))&end=$NOW&bucket=3600"
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Stream error: Could not open …" | Wrong URL or path, RTSP disabled on the camera, a password that needs URL-encoding, or the camera unreachable from the Docker host. The counter retries automatically |
| Status flips between Streaming and Stream error | The stream stalls for more than 10 seconds; check the network path to the camera |
| High CPU use | Pointed at a high-resolution main stream; switch to the sub stream, or lower the analysis rate |
| Analysis rate below the configured value | The CPU cannot keep up; lower the analysis rate or raise `THREADS` if cores are free |
| The same person appears twice | Lower the match strictness or lengthen the quick-return and re-entry windows |
| Different people merged into one | Raise the match strictness or shorten the windows |
| Fast movers are not counted | Raise the analysis rate, or lower "Sightings before counting" |
| Objects counted as people | Raise the detection confidence |
| Changes to `.env` have no effect | Settings saved in the UI override it; see the note under [Settings](#settings) |

Container logs:

```bash
docker compose logs -f
```

## Project layout

| Path | Purpose |
|---|---|
| `app/engine.py` | Capture, detection, tracking, and the counting and merging logic |
| `app/store.py` | SQLite storage |
| `app/config.py` | Settings: defaults, validation, persistence |
| `app/main.py` | Web server and API |
| `app/static/index.html` | The dashboard, a single self-contained page |
| `Dockerfile`, `docker-compose.yml` | Container build and run configuration |

## Licence

AGPL-3.0-or-later; see [LICENSE](LICENSE). Detection uses
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0,
and the YOLO11n weights are downloaded from Ultralytics when the image is built.
