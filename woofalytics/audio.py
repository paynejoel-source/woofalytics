from __future__ import annotations

import logging
import subprocess

import numpy as np

try:
    import pyaudio
except ImportError as exc:  # pragma: no cover - exercised by setup diagnostics
    pyaudio = None
    PYAUDIO_IMPORT_ERROR = exc
else:
    PYAUDIO_IMPORT_ERROR = None


class AudioCapture:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_samples: int,
        device_name_hint: str = "",
        input_device_index: int | None = None,
    ):
        if pyaudio is None:
            raise RuntimeError(
                f"PyAudio is unavailable: {PYAUDIO_IMPORT_ERROR}. Install runtime dependencies first."
            )

        self._logger = logging.getLogger("Woofalytics.Audio")
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = chunk_samples
        self._device_name_hint = device_name_hint
        self._requested_input_device_index = input_device_index
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._input_device_index = self._resolve_device_index()

    def _resolve_device_index(self) -> int | None:
        if self._requested_input_device_index is not None:
            return self._requested_input_device_index

        host_api = self._pa.get_host_api_info_by_index(0)
        device_count = host_api.get("deviceCount", 0)
        hint = self._device_name_hint.strip().lower()

        fallback = None
        for idx in range(device_count):
            info = self._pa.get_device_info_by_index(idx)
            max_input_channels = int(info.get("maxInputChannels", 0))
            if max_input_channels <= 0:
                continue

            name = str(info.get("name", ""))
            self._logger.debug("Input device %s: %s", idx, name)
            if fallback is None:
                fallback = idx
            if hint and hint in name.lower():
                self._logger.info("Selected input device %s: %s", idx, name)
                return idx

        if hint:
            raise RuntimeError(
                f"No input device matched {self._device_name_hint!r}. Set WOOF_INPUT_DEVICE_INDEX or adjust WOOF_DEVICE_NAME_HINT."
            )

        if fallback is None:
            raise RuntimeError("No audio input device is available.")

        info = self._pa.get_device_info_by_index(fallback)
        self._logger.info(
            "Using default input device %s: %s", fallback, info.get("name", "unknown")
        )
        return fallback

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            frames_per_buffer=self._chunk_samples,
            input=True,
            input_device_index=self._input_device_index,
        )

    def read_chunk(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Audio capture has not been started.")

        data = self._stream.read(self._chunk_samples, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        if self._channels > 1:
            audio = audio.reshape((-1, self._channels)).mean(axis=1).astype(np.int16)
        return audio

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()


class FFmpegAudioCapture:
    def __init__(
        self,
        stream_url: str,
        ffmpeg_path: str,
        sample_rate: int,
        channels: int,
        chunk_samples: int,
    ):
        self._logger = logging.getLogger("Woofalytics.FFmpeg")
        self._stream_url = stream_url
        self._ffmpeg_path = ffmpeg_path
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = chunk_samples
        self._process = None

    def start(self) -> None:
        if self._process is not None:
            return

        command = [
            self._ffmpeg_path,
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-allowed_media_types",
            "audio",
            "-i",
            self._stream_url,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(self._channels),
            "-ar",
            str(self._sample_rate),
            "-f",
            "s16le",
            "-",
        ]
        self._logger.info("Starting ffmpeg audio ingest from %s", self._stream_url)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg was not found at {self._ffmpeg_path}. Set WOOF_FFMPEG_PATH to a valid binary."
            ) from exc

    def read_chunk(self) -> np.ndarray:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("FFmpeg audio capture has not been started.")

        frame_bytes = self._chunk_samples * self._channels * 2
        data = self._process.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read1(4096)
            detail = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                "ffmpeg audio stream ended unexpectedly."
                + (f" Details: {detail}" if detail else "")
            )

        audio = np.frombuffer(data, dtype=np.int16)
        if self._channels > 1:
            audio = audio.reshape((-1, self._channels)).mean(axis=1).astype(np.int16)
        return audio

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
        self._process = None
