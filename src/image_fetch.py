import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

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


_fetch_client = httpx.AsyncClient(
    follow_redirects=False,
    timeout=None,
)


async def aclose_client() -> None:
    await _fetch_client.aclose()


async def _resolve_public_ips(host: str) -> list[str]:
    loop = asyncio.get_event_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ImageFetchError(f"DNS resolution failed: {e}", transient=True)
    ips: list[str] = []
    for info in infos:
        addr = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not _is_public_ip(ip_obj):
            raise ImageFetchError(
                f"resolves to non-public address {addr}", transient=False
            )
        ips.append(addr)
    if not ips:
        raise ImageFetchError("host did not resolve", transient=True)
    return ips


async def _fetch_bytes(url: str) -> tuple[bytes, str]:
    try:
        async with asyncio.timeout(FETCH_TIMEOUT):
            return await _fetch_loop(url)
    except TimeoutError:
        raise ImageFetchError("timed out", transient=True)
    except httpx.RequestError as e:
        raise ImageFetchError(f"request failed: {type(e).__name__}", transient=True)
    except httpx.InvalidURL as e:
        # httpx builds the redirect URL eagerly (even with follow_redirects=False) to
        # populate response.next_request, and raises this for malformed Location values
        # (e.g. a non-http(s) scheme like data:/file:) before our own scheme check runs.
        raise ImageFetchError(f"invalid redirect target: {e}", transient=False)


async def _fetch_loop(url: str) -> tuple[bytes, str]:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise ImageFetchError(
                f"unsupported scheme {parsed.scheme!r}", transient=False
            )
        host = parsed.hostname
        if not host:
            raise ImageFetchError("missing host", transient=False)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        ips = await _resolve_public_ips(host)
        ip = ips[0]
        ip_authority = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
        target = urlunparse(
            (parsed.scheme, ip_authority, parsed.path or "/", parsed.params, parsed.query, "")
        )
        host_hdr = f"[{host}]" if ":" in host else host
        if port not in (80, 443):
            host_hdr = f"{host_hdr}:{port}"
        headers = {"Host": host_hdr, "User-Agent": USER_AGENT, "Accept": "image/*"}

        async with _fetch_client.stream(
            "GET", target, headers=headers, extensions={"sni_hostname": host}
        ) as resp:
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if not loc:
                    raise ImageFetchError(
                        f"redirect {resp.status_code} without location", transient=False
                    )
                current = urljoin(current, loc)
                continue
            if resp.status_code >= 400:
                raise ImageFetchError(
                    f"HTTP {resp.status_code}", transient=resp.status_code >= 500
                )
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) > MAX_IMAGE_BYTES:
                    raise ImageFetchError(
                        f"exceeds {MAX_IMAGE_BYTES} bytes", transient=False
                    )
            mime = _sniff_mime(bytes(buf))
            if mime is None:
                raise ImageFetchError("not a recognized image", transient=False)
            return bytes(buf), mime
    raise ImageFetchError(f"too many redirects (>{MAX_REDIRECTS})", transient=False)
