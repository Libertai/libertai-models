import base64
import io
import threading
from typing import Optional

import torch
from diffusers import ZImagePipeline
from PIL import Image


class ImagePipelineManager:
    """Singleton managing loaded Z-Image pipeline with thread-safe loading"""

    _instance: Optional["ImagePipelineManager"] = None
    _pipeline: Optional[ZImagePipeline] = None
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_pipeline(self, model_path: str) -> None:
        """Load Z-Image pipeline if not already loaded (thread-safe)"""
        if self._pipeline is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._pipeline is None:
                    dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
                    print(f"[Image Generation] Loading model: {model_path}")
                    print(f"[Image Generation] Device: {self._device}, dtype: {dtype}")

                    try:
                        self._pipeline = ZImagePipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                        )
                        self._pipeline.to(self._device)  # type: ignore[union-attr]
                        print("[Image Generation] Model loaded and ready!")
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[Image Generation] CUDA OOM during model loading: {e}")
                        raise RuntimeError("GPU out of memory. Cannot load image generation model.") from e
                    except Exception as e:
                        print(f"[Image Generation] Failed to load model: {e}")
                        raise RuntimeError(f"Failed to load image generation model: {str(e)}") from e

    def get_pipeline(self) -> ZImagePipeline:
        """Get loaded pipeline"""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        return self._pipeline


def generate_image(
    pipeline: ZImagePipeline,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    guidance_scale: float = 0.0,
    seed: int = -1,
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

    Returns:
        Base64-encoded PNG image string
    """
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

        # Convert to base64 PNG using context manager
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return image_b64

    except torch.cuda.OutOfMemoryError as e:
        print(f"[Image Generation] CUDA OOM during generation: {e}")
        raise RuntimeError("GPU out of memory during image generation.") from e
    finally:
        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
