import ipaddress
from dataclasses import dataclass

FETCH_TIMEOUT = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 3
MAX_IMAGES_PER_REQUEST = 4
GLOBAL_FETCH_CONCURRENCY = 16
CACHE_MAX_BYTES = 64 * 1024 * 1024
CACHE_TTL = 600
NEG_CACHE_TTL_DETERMINISTIC = 60
NEG_CACHE_TTL_TRANSIENT = 5
USER_AGENT = "LibertAI/1.0 (+https://libertai.io)"

IMAGE_INLINE_PATHS = {"v1/chat/completions", "v1/messages", "v1/responses"}


class ImageFetchError(Exception):
    """A recoverable, user-facing fetch/validation failure (maps to HTTP 400)."""

    def __init__(self, reason: str, *, transient: bool):
        super().__init__(reason)
        self.reason = reason
        self.transient = transient


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Python 3.11.2 (pre-3.11.9) misclassifies IPv4-mapped IPv6 as global; unwrap first.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def _sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass
class _ImagePart:
    part: dict
    kind: str  # "openai_chat" | "openai_responses" | "anthropic"
    url: str


def _collect_image_parts(body_json) -> list["_ImagePart"]:
    out: list[_ImagePart] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            t = obj.get("type")
            if t == "image_url" and isinstance(obj.get("image_url"), dict):
                u = obj["image_url"].get("url")
                if isinstance(u, str) and not u.startswith("data:"):
                    out.append(_ImagePart(obj, "openai_chat", u))
            elif t == "input_image" and isinstance(obj.get("image_url"), str):
                u = obj["image_url"]
                if not u.startswith("data:"):
                    out.append(_ImagePart(obj, "openai_responses", u))
            elif t == "image" and isinstance(obj.get("source"), dict) \
                    and obj["source"].get("type") == "url":
                u = obj["source"].get("url")
                if isinstance(u, str) and not u.startswith("data:"):
                    out.append(_ImagePart(obj, "anthropic", u))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(body_json)
    return out


def _rewrite(part: "_ImagePart", b64: str, mime: str) -> None:
    data_url = f"data:{mime};base64,{b64}"
    if part.kind == "openai_chat":
        part.part["image_url"]["url"] = data_url
    elif part.kind == "openai_responses":
        part.part["image_url"] = data_url
    elif part.kind == "anthropic":
        part.part["source"] = {"type": "base64", "media_type": mime, "data": b64}
