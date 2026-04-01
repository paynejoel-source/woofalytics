# Woofalytics

Original project credit belongs to [mdoulaty](https://github.com/mdoulaty), the original author of Woofalytics. This repository is a derivative continuation of that work.

Woofalytics is now a small dog-first sound monitor built around a pretrained audio-event model instead of a custom training prototype.

The scope is intentionally narrow:

- detect barking in live audio
- detect thunder in live audio
- detect train whistles and speech for context
- save clips around triggered sound events
- expose a simple local dashboard and JSON API
- export sound events as CSV
- optionally trigger IFTTT when sound events fire

It does not try to classify breed, identity, or emotion.

## Architecture

The runtime is built around a YAMNet-style detector:

- live audio capture from a Frigate/go2rtc RTSP restream, using the substream by default
- LiteRT (`ai-edge-litert`) for lightweight TFLite inference
- sliding inference window at 16 kHz mono
- pretrained YAMNet class thresholding with a dynamic sound-rule stack
- cooldown gate to avoid trigger spam
- clip capture with pre-roll and post-roll audio
- local HTTP dashboard and `/api/status` endpoint
- CSV export at `/api/events.csv`

## Install

For package-style installation from the repo root:

```shell
python -m pip install .
```

For an editable local development install:

```shell
python -m pip install -e .
```

## Dependencies

Install system audio packages first:

```shell
sudo apt update
sudo apt install build-essential libportaudio2 libasound2-dev python3-pyaudio
```

`ffmpeg` is also required for RTSP ingest. A typical local go2rtc/Frigate audio source looks like:

```text
rtsp://127.0.0.1:8554/camera_audio
```

Woofalytics should use a substream by default so it does not share the main recording path:

```text
rtsp://127.0.0.1:8554/camera_audio_sub
```

Create a virtualenv if you do not already have one:

```shell
/usr/bin/python3 -m venv venv
source venv/bin/activate
```

Download the pretrained YAMNet TFLite model:

```shell
woofalytics-download-model
```

Run a quick diagnostic:

```shell
woofalytics-check-setup
```

If you only want to review the UI or smoke-test the app path without audio hardware, run demo mode:

```shell
WOOF_DEMO_MODE=1 woofalytics
```

## Running

Start the monitor:

```shell
woofalytics
```

The repo still includes `main.py` and the `scripts/` wrappers for compatibility with the current local deployment.

Open the dashboard at `http://127.0.0.1:8015` if you use the Desktop launcher, or at the port you set in `WOOF_PORT`.

The desktop launcher now installs and uses a user `systemd` service at:

```text
~/.config/systemd/user/woofalytics.service
```

That service is configured to restart automatically if the Python process exits.

## Configuration

The runtime is configured with environment variables.

Common ones:

- `WOOF_PORT=8015`
- `WOOF_AUDIO_SOURCE=ffmpeg`
- `WOOF_STREAM_URL=rtsp://127.0.0.1:8554/camera_audio_sub`
- `WOOF_FFMPEG_PATH=/usr/bin/ffmpeg`
- `WOOF_MODEL_PATH=./models/yamnet.tflite`
- `WOOF_EVENTS_CSV_PATH=./events/events.csv`
- `WOOF_BARK_THRESHOLD=0.55`
- `WOOF_THUNDER_THRESHOLD=0.65`
- `WOOF_TRAIN_WHISTLE_THRESHOLD=0.60`
- `WOOF_SPEECH_THRESHOLD=0.75`
- `WOOF_CLIP_PRE_SECONDS=8`
- `WOOF_CLIP_POST_SECONDS=8`
- `WOOF_CLIP_RETENTION_DAYS=30`
- `WOOF_TRIGGER_COOLDOWN_SECONDS=8`
- `WOOF_CAPTURE_STALL_SECONDS=15`
- `WOOF_EXTRA_SOUNDS_JSON=[{"label":"Chainsaw","threshold":0.7,"target_indices":[390],"target_labels":{"390":"chainsaw"}}]`
- `WOOF_DEVICE_NAME_HINT=USB audio device`
- `WOOF_INPUT_DEVICE_INDEX=2`
- `WOOF_IFTTT_EVENT_NAME=woof`
- `WOOF_IFTTT_KEY=...`
- `WOOF_DEMO_MODE=1`

Built-in rules currently ship in this order:

- bark
- thunder
- train whistle
- speech

Extra rules can be injected at startup with `WOOF_EXTRA_SOUNDS_JSON`. Each item accepts:

- `label`
- `key` (optional, otherwise derived from `label`)
- `threshold`
- `target_indices`
- `target_labels`

All configured rules are exposed dynamically in the dashboard and `GET /api/status`, and any rule that crosses threshold can create a full audio clip.

Clips and analytics are retained separately:

- clip `.wav` files are pruned by age using `WOOF_CLIP_RETENTION_DAYS`
- the CSV analytics log is cumulative and is not trimmed by clip retention

## API

`GET /api/status`

Returns the latest sound stack, thresholds, current event flag, capture health fields, target scores, and recent events.

`POST /api/record`

Queues a manual clip capture using the current pre-roll and post-roll settings.

`GET /api/events.csv`

Downloads the accumulated sound event log as CSV.

## Repo Cleanup

The repo is now split more cleanly between:

- the installable Python package under `woofalytics/`
- optional local deployment assets under `assets/`
- compatibility wrappers under `main.py` and `scripts/`

The old Torch model files, notebook workflow, and handwritten direction-of-arrival code are no longer part of the runtime design. If you still want a research or training workflow later, it should live in a separate training package instead of the production detector.
