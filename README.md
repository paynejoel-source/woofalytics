# Woofalytics

Woofalytics is now a small dog-first sound monitor built around a pretrained audio-event model instead of a custom training prototype.

The scope is intentionally narrow:

- detect barking in live audio
- detect thunder in live audio
- save clips around bark and thunder events
- expose a simple local dashboard and JSON API
- export bark and thunder events as CSV
- optionally trigger IFTTT when sound events fire

It does not try to classify breed, identity, or emotion.

## Architecture

The runtime is built around a YAMNet-style detector:

- live audio capture from the same PoE camera RTSP stream used by BirdNET-Go
- LiteRT (`ai-edge-litert`) for lightweight TFLite inference
- sliding inference window at 16 kHz mono
- pretrained bark and thunder class score thresholding
- cooldown gate to avoid trigger spam
- clip capture with pre-roll and post-roll audio
- local HTTP dashboard and `/api/status` endpoint
- CSV export at `/api/events.csv`

## Dependencies

Install system audio packages first:

```shell
sudo apt update
sudo apt install build-essential libportaudio2 libasound2-dev python3-pyaudio
```

`ffmpeg` is also required for RTSP ingest. BirdNET-Go is already configured on this machine to use:

```text
rtsp://127.0.0.1:8554/front_yard
```

Create a virtualenv:

```shell
/usr/bin/python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```shell
pip install -r requirements.txt
```

Download the pretrained YAMNet TFLite model:

```shell
python scripts/download_yamnet.py
```

Run a quick diagnostic:

```shell
python scripts/check_setup.py
```

If you only want to review the UI or smoke-test the app path without audio hardware, run demo mode:

```shell
WOOF_DEMO_MODE=1 python main.py
```

## Running

Start the monitor:

```shell
python main.py
```

Open the dashboard at `http://127.0.0.1:8015` if you use the Desktop launcher, or at the port you set in `WOOF_PORT`.

## Configuration

The runtime is configured with environment variables.

Common ones:

- `WOOF_PORT=8015`
- `WOOF_AUDIO_SOURCE=ffmpeg`
- `WOOF_STREAM_URL=rtsp://127.0.0.1:8554/front_yard`
- `WOOF_FFMPEG_PATH=/usr/bin/ffmpeg`
- `WOOF_MODEL_PATH=./models/yamnet.tflite`
- `WOOF_EVENTS_CSV_PATH=./events/events.csv`
- `WOOF_BARK_THRESHOLD=0.55`
- `WOOF_THUNDER_THRESHOLD=0.55`
- `WOOF_CLIP_PRE_SECONDS=8`
- `WOOF_CLIP_POST_SECONDS=8`
- `WOOF_TRIGGER_COOLDOWN_SECONDS=8`
- `WOOF_DEVICE_NAME_HINT=Andrea PureAudio`
- `WOOF_INPUT_DEVICE_INDEX=2`
- `WOOF_IFTTT_EVENT_NAME=woof`
- `WOOF_IFTTT_KEY=...`
- `WOOF_DEMO_MODE=1`

## API

`GET /api/status`

Returns the latest bark score, thunder score, thresholds, current event flag, target scores, and recent events.

`POST /api/record`

Queues a manual clip capture using the current pre-roll and post-roll settings.

`GET /api/events.csv`

Downloads the accumulated bark and thunder event log as CSV.

## Repo Cleanup

The old Torch model files, notebook workflow, and handwritten direction-of-arrival code are no longer part of the runtime design. If you still want a research or training workflow later, it should live in a separate training package instead of the production detector.
