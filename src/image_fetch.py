import asyncio
import base64
import ipaddress
import logging
import socket
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from fastapi import HTTPException

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


# No keep-alive: connections are pinned to a validated IP, so a pooled TLS
# session for one hostname must not be reused for another host on the same IP.
_fetch_client = httpx.AsyncClient(
    follow_redirects=False,
    timeout=None,
    limits=httpx.Limits(max_keepalive_connections=0),
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


logger = logging.getLogger(__name__)

_positive: "OrderedDict[str, tuple[float, str, str]]" = OrderedDict()  # url -> (expiry, b64, mime)
_positive_bytes = 0
_negative: dict[str, tuple[float, str]] = {}  # url -> (expiry, reason)
_inflight: dict[str, asyncio.Future] = {}
_cache_lock = asyncio.Lock()
_fetch_semaphore = asyncio.Semaphore(GLOBAL_FETCH_CONCURRENCY)

_stats = {"hits": 0, "misses": 0, "evictions": 0, "neg_hits": 0, "coalesced": 0}


def _reset_cache_for_tests() -> None:
    global _positive_bytes
    _positive.clear()
    _negative.clear()
    _inflight.clear()
    _positive_bytes = 0
    for k in _stats:
        _stats[k] = 0


def _cache_put(url: str, b64: str, mime: str) -> None:
    global _positive_bytes
    size = len(b64)
    now = asyncio.get_event_loop().time()
    old = _positive.pop(url, None)
    if old is not None:
        _positive_bytes -= len(old[1])
    _positive[url] = (now + CACHE_TTL, b64, mime)
    _positive.move_to_end(url)
    _positive_bytes += size
    while _positive_bytes > CACHE_MAX_BYTES and _positive:
        _, (_, old_b64, _m) = _positive.popitem(last=False)
        _positive_bytes -= len(old_b64)
        _stats["evictions"] += 1


async def get_or_fetch(url: str) -> tuple[str, str]:
    now = asyncio.get_event_loop().time()
    async with _cache_lock:
        ent = _positive.get(url)
        if ent and ent[0] > now:
            _positive.move_to_end(url)
            _stats["hits"] += 1
            return ent[1], ent[2]
        neg = _negative.get(url)
        if neg and neg[0] > now:
            _stats["neg_hits"] += 1
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Failed to fetch image URL {url}: {neg[1]}",
            )
        fut = _inflight.get(url)
        if fut is None:
            _stats["misses"] += 1
            fut = asyncio.get_event_loop().create_future()
            _inflight[url] = fut
            asyncio.create_task(_run_fetch(url, fut))
        else:
            _stats["coalesced"] += 1
    return await asyncio.shield(fut)


async def _run_fetch(url: str, fut: asyncio.Future) -> None:
    try:
        async with _fetch_semaphore:
            raw, mime = await _fetch_bytes(url)
        b64 = base64.b64encode(raw).decode()
    except ImageFetchError as e:
        ttl = NEG_CACHE_TTL_TRANSIENT if e.transient else NEG_CACHE_TTL_DETERMINISTIC
        async with _cache_lock:
            _inflight.pop(url, None)
            now_t = asyncio.get_event_loop().time()
            for k in [k for k, (exp, _r) in _negative.items() if exp <= now_t]:
                del _negative[k]
            _negative[url] = (now_t + ttl, e.reason)
        logger.warning("image fetch failed url=%s reason=%s transient=%s", url, e.reason, e.transient)
        if not fut.done():
            fut.set_exception(
                HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Failed to fetch image URL {url}: {e.reason}",
                )
            )
        return
    except Exception as e:  # unexpected — do not poison cache long-term
        async with _cache_lock:
            _inflight.pop(url, None)
        logger.exception("unexpected image fetch error url=%s", url)
        if not fut.done():
            fut.set_exception(
                HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Failed to fetch image URL {url}: {type(e).__name__}",
                )
            )
        return
    async with _cache_lock:
        _inflight.pop(url, None)
        _cache_put(url, b64, mime)
    if not fut.done():
        fut.set_result((b64, mime))


async def inline_remote_images(full_path: str, body_json: dict) -> tuple[dict, bool]:
    parts = _collect_image_parts(body_json)
    if not parts:
        return body_json, False

    unique = list({p.url for p in parts})
    if len(unique) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Too many image URLs in request (max {MAX_IMAGES_PER_REQUEST})",
        )

    tasks = {u: asyncio.ensure_future(get_or_fetch(u)) for u in unique}
    try:
        await asyncio.gather(*tasks.values())
    except BaseException:
        for t in tasks.values():
            t.cancel()
        raise
    results = {u: t.result() for u, t in tasks.items()}

    for p in parts:
        b64, mime = results[p.url]
        _rewrite(p, b64, mime)
    return body_json, True
