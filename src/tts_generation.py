import io
import threading
from typing import Any, Optional

try:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline

    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    np = None  # type: ignore
    sf = None  # type: ignore
    torch = None  # type: ignore
    KPipeline = None  # type: ignore


class TTSModelManager:
    """Singleton managing Kokoro TTS pipelines with on-demand loading.

    Mirrors ImageModelManager: refcounted load, per-model inference lock, OOM eviction.
    Kokoro pipelines are not guaranteed thread-safe, so the inference lock serializes
    synthesis per model.
    """

    _instance: Optional["TTSModelManager"] = None
    _pipelines: dict[str, Any] = {}
    _refcounts: dict[str, int] = {}
    _model_configs: dict[str, Any] = {}  # AudioModelConfig
    _inference_locks: dict[str, threading.Lock] = {}
    _device: str = "cuda" if KOKORO_AVAILABLE and torch.cuda.is_available() else "cpu"
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TTSModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
            cls._instance._refcounts = {}
            cls._instance._model_configs = {}
            cls._instance._inference_locks = {}
        return cls._instance

    def register(self, model_id: str, model_config: Any) -> None:
        self._model_configs[model_id] = model_config

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._pipelines

    def is_capable(self, model_id: str) -> bool:
        return model_id in self._model_configs

    def _load(self, model_id: str) -> None:
        cfg = self._model_configs[model_id]
        self._pipelines[model_id] = KPipeline(lang_code=cfg.lang_code, repo_id=cfg.local_path)

    def _unload_all_except(self, keep_model_id: str) -> None:
        for mid in list(self._pipelines.keys()):
            if mid == keep_model_id:
                continue
            if self._refcounts.get(mid, 0) > 0:
                continue
            del self._pipelines[mid]
        if KOKORO_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def acquire(self, model_id: str) -> tuple[Any, threading.Lock]:
        """Get (pipeline, inference_lock) with refcount increment.

        Caller MUST call release() when done and hold the inference_lock during synthesis.
        """
        with self._lock:
            if model_id not in self._model_configs:
                raise RuntimeError(f"Model '{model_id}' not registered")
            if model_id not in self._pipelines:
                try:
                    self._load(model_id)
                except Exception as e:
                    if "out of memory" in str(e).lower() or (
                        KOKORO_AVAILABLE and isinstance(e, torch.cuda.OutOfMemoryError)
                    ):
                        self._unload_all_except(model_id)
                        self._load(model_id)
                    else:
                        raise
            if model_id not in self._inference_locks:
                self._inference_locks[model_id] = threading.Lock()
            self._refcounts[model_id] = self._refcounts.get(model_id, 0) + 1
            return self._pipelines[model_id], self._inference_locks[model_id]

    def release(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._refcounts:
                self._refcounts[model_id] = max(0, self._refcounts[model_id] - 1)


SAMPLE_RATE = 24000


def synthesize_wav(pipeline: Any, text: str, voice: str, speed: float) -> bytes:
    """Run Kokoro and return WAV bytes (24kHz mono).

    Verified shape (Phase 0, kokoro 0.9.4): each yielded Result is
    (graphemes: str, phonemes: str, audio: torch.Tensor) at 24kHz.
    """
    segments = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
        segments.append(np.asarray(audio, dtype=np.float32))
    if not segments:
        raise RuntimeError("Kokoro produced no audio")
    waveform = np.concatenate(segments)
    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()
