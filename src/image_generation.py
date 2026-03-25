import base64
import io
import threading
from typing import Any, Optional

try:
    import torch
    from diffusers import ZImagePipeline
    from PIL import Image

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    torch = None  # type: ignore
    ZImagePipeline = None  # type: ignore
    Image = None  # type: ignore

# Optional: background removal
try:
    from rembg import remove as remove_bg, new_session  # type: ignore

    # Force CPU for rembg to avoid GPU memory conflicts with the main model
    REMBG_SESSION = new_session("u2net", providers=["CPUExecutionProvider"])
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    REMBG_SESSION = None
    print("[Image Generation] rembg not installed, background removal disabled")


class ImageModelManager:
    """Singleton managing image pipelines (generation + editing) with on-demand loading."""

    _instance: Optional["ImageModelManager"] = None
    _pipelines: dict[str, Any] = {}
    _refcounts: dict[str, int] = {}
    _model_configs: dict[str, Any] = {}  # ImageModelConfig | ImageEditModelConfig
    _device: str = "cuda" if DIFFUSERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ImageModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
            cls._instance._refcounts = {}
            cls._instance._model_configs = {}
        return cls._instance

    def register(self, model_id: str, model_config: Any) -> None:
        """Register a model config at startup."""
        self._model_configs[model_id] = model_config

    def is_loaded(self, model_id: str) -> bool:
        """True if pipeline is currently in memory."""
        return model_id in self._pipelines

    def is_capable(self, model_id: str) -> bool:
        """True if model config is registered."""
        return model_id in self._model_configs

    def acquire(self, model_id: str) -> Any:
        """Get pipeline with refcount increment. Loads on demand. OOM evicts others."""
        with self._lock:
            if model_id not in self._model_configs:
                raise RuntimeError(f"Model '{model_id}' not registered")

            if model_id not in self._pipelines:
                try:
                    self._load(model_id)
                except Exception as e:
                    if "out of memory" in str(e).lower() or (
                        DIFFUSERS_AVAILABLE and isinstance(e, torch.cuda.OutOfMemoryError)
                    ):
                        self._unload_all_except(model_id)
                        self._load(model_id)  # Retry once; if still OOM, let it raise
                    else:
                        raise

            self._refcounts[model_id] = self._refcounts.get(model_id, 0) + 1
            return self._pipelines[model_id]

    def release(self, model_id: str) -> None:
        """Decrement refcount after pipeline use."""
        with self._lock:
            if model_id in self._refcounts:
                self._refcounts[model_id] = max(0, self._refcounts[model_id] - 1)

    def _load(self, model_id: str) -> None:
        """Load the appropriate pipeline based on config type."""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "Image dependencies not installed. "
                "Install with: pip install -e '.[image]' or poetry install --extras image"
            )

        from src.config import ImageEditModelConfig

        model_config = self._model_configs[model_id]
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        print(f"[ImageModelManager] Loading {model_id} from {model_config.local_path}")
        print(f"[ImageModelManager] Device: {self._device}, dtype: {dtype}")

        try:
            if isinstance(model_config, ImageEditModelConfig):
                from diffusers import QwenImageEditPlusPipeline

                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    model_config.local_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                pipeline = ZImagePipeline.from_pretrained(
                    model_config.local_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            pipeline.to(self._device)
            self._pipelines[model_id] = pipeline
            self._refcounts[model_id] = 0
            print(f"[ImageModelManager] {model_id} loaded and ready!")
        except torch.cuda.OutOfMemoryError as e:
            print(f"[ImageModelManager] CUDA OOM loading {model_id}: {e}")
            raise
        except Exception as e:
            print(f"[ImageModelManager] Failed to load {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

    def _unload_all_except(self, keep_model_id: str) -> None:
        """Unload all pipelines except the one we want to load. Skips in-use pipelines."""
        for mid in list(self._pipelines.keys()):
            if mid == keep_model_id:
                continue
            if self._refcounts.get(mid, 0) > 0:
                print(f"[ImageModelManager] Skipping unload of {mid} (refcount={self._refcounts[mid]})")
                continue
            print(f"[ImageModelManager] Unloading {mid} to free memory")
            pipeline = self._pipelines.pop(mid)
            if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
                pipeline.to("cpu")
            del pipeline
            self._refcounts.pop(mid, None)

        if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_image(
    pipeline: "ZImagePipeline",  # type: ignore
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    guidance_scale: float = 0.0,
    seed: int = -1,
    remove_background: bool = False,
) -> str:
    """
    Generate image and return base64-encoded PNG.

    Args:
        pipeline: Z-Image pipeline instance
        prompt: Text prompt for image generation
        width: Image width
        height: Image height
        steps: Number of inference steps (Z-Image-Turbo works well with 8-9)
        guidance_scale: CFG scale (Turbo model doesn't need CFG, use 0.0)
        seed: Random seed (-1 for random)
        remove_background: Remove background with rembg (requires rembg installed)

    Returns:
        Base64-encoded PNG image string
    """
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError(
            "Image generation dependencies not installed. "
            "Install with: pip install -e '.[image]' or poetry install --extras image"
        )

    try:
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)  # type: ignore[attr-defined]

        result = pipeline(  # type: ignore[operator]
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image: Image.Image = result.images[0]

        # Remove background if requested
        if remove_background:
            if REMBG_AVAILABLE and REMBG_SESSION:
                image = remove_bg(
                    image,
                    session=REMBG_SESSION,
                )
            else:
                print("[Image Generation] Warning: remove_background requested but rembg not available")

        # Convert to base64 PNG using context manager (supports transparency)
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return image_b64

    except torch.cuda.OutOfMemoryError as e:  # type: ignore
        print(f"[Image Generation] CUDA OOM during generation: {e}")
        raise RuntimeError("GPU out of memory during image generation.") from e
    finally:
        # Clear CUDA cache to prevent memory accumulation
        if DIFFUSERS_AVAILABLE and torch.cuda.is_available():  # type: ignore
            torch.cuda.empty_cache()  # type: ignore


def edit_image(
    pipeline: Any,
    images: list["Image.Image"],
    prompt: str,
    num_images: int = 1,
    seed: int = -1,
) -> list[str]:
    """
    Edit image(s) and return list of base64-encoded PNGs.

    Args:
        pipeline: QwenImageEditPlusPipeline instance
        images: Input images to edit
        prompt: Edit instruction
        num_images: Number of output images
        seed: Random seed (-1 for random)

    Returns:
        List of base64-encoded PNG image strings
    """
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError(
            "Image dependencies not installed. "
            "Install with: pip install -e '.[image]' or poetry install --extras image"
        )

    try:
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        result = pipeline(
            image=images,
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=40,
            guidance_scale=1.0,
            true_cfg_scale=4.0,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        output_images = []
        for img in result.images:
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                output_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return output_images

    except torch.cuda.OutOfMemoryError as e:
        print(f"[Image Edit] CUDA OOM during editing: {e}")
        raise RuntimeError("GPU out of memory during image editing.") from e
    finally:
        if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
