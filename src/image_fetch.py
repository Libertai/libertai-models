import ipaddress

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
