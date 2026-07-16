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
NEG_CACHE_TTL_UNEXPECTED = 1
MAX_NEG_ENTRIES = 1024
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
    # Iterative walk (explicit stack) so a deeply nested body can't blow the
    # Python recursion limit — json.loads accepts far deeper nesting than
    # recursion would survive.
    out: list[_ImagePart] = []
    stack = [body_json]
    while stack:
        obj = stack.pop()
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
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)
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
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        # UnicodeError: IDNA encoding of a malformed hostname.
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
        try:
            parsed = urlparse(current)
            scheme, host, port = parsed.scheme, parsed.hostname, parsed.port
        except ValueError:
            # urlparse().port raises on an out-of-range/non-numeric port.
            raise ImageFetchError("malformed URL", transient=False)
        if scheme not in ("http", "https"):
            raise ImageFetchError(f"unsupported scheme {scheme!r}", transient=False)
        if not host:
            raise ImageFetchError("missing host", transient=False)
        default_port = 443 if scheme == "https" else 80
        port = port or default_port

        ips = await _resolve_public_ips(host)
        ip = ips[0]
        ip_authority = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
        target = urlunparse(
            (scheme, ip_authority, parsed.path or "/", parsed.params, parsed.query, "")
        )
        host_hdr = f"[{host}]" if ":" in host else host
        if port != default_port:
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
                # 5xx and 429 (rate limited — the original wikimedia failure mode)
                # are retry-later: short negative TTL. Other 4xx are deterministic.
                transient = resp.status_code >= 500 or resp.status_code == 429
                raise ImageFetchError(f"HTTP {resp.status_code}", transient=transient)
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) > MAX_IMAGE_BYTES:
                    raise ImageFetchError(
                        f"exceeds {MAX_IMAGE_BYTES} bytes", transient=False
                    )
            mime = _sniff_mime(bytes(buf[:16]))  # magic bytes only — avoid a full copy
            if mime is None:
                raise ImageFetchError("not a recognized image", transient=False)
            return bytes(buf), mime
    raise ImageFetchError(f"too many redirects (>{MAX_REDIRECTS})", transient=False)


logger = logging.getLogger(__name__)

_positive: "OrderedDict[str, tuple[float, str, str]]" = OrderedDict()  # url -> (expiry, b64, mime)
_positive_bytes = 0
_negative: "OrderedDict[str, tuple[float, str]]" = OrderedDict()  # url -> (expiry, reason)
_inflight: dict[str, asyncio.Future] = {}
_cache_lock = asyncio.Lock()
_fetch_semaphore = asyncio.Semaphore(GLOBAL_FETCH_CONCURRENCY)
# Hold strong refs to detached fetch tasks; the loop only keeps weak ones, so
# without this a task could be garbage-collected mid-flight (CPython docs).
_background_tasks: "set[asyncio.Task]" = set()


def _reset_cache_for_tests() -> None:
    global _positive_bytes
    _positive.clear()
    _negative.clear()
    _inflight.clear()
    _positive_bytes = 0


def _cache_put(url: str, b64: str, mime: str) -> None:
    global _positive_bytes
    size = len(b64)
    now = asyncio.get_running_loop().time()
    # Drop expired entries so their bytes don't hold capacity against live ones
    # (mirrors _cache_negative's sweep).
    for k in [k for k, (exp, _b, _m) in _positive.items() if exp <= now]:
        _positive_bytes -= len(_positive[k][1])
        del _positive[k]
    old = _positive.pop(url, None)
    if old is not None:
        _positive_bytes -= len(old[1])
    _positive[url] = (now + CACHE_TTL, b64, mime)
    _positive.move_to_end(url)
    _positive_bytes += size
    while _positive_bytes > CACHE_MAX_BYTES and _positive:
        _, (_, old_b64, _m) = _positive.popitem(last=False)
        _positive_bytes -= len(old_b64)


def _cache_negative(url: str, ttl: float, reason: str) -> None:
    # Purge expired entries, then bound the dict (FIFO) so a flood of distinct
    # failing URLs can't grow it without limit.
    now = asyncio.get_running_loop().time()
    for k in [k for k, (exp, _r) in _negative.items() if exp <= now]:
        del _negative[k]
    _negative[url] = (now + ttl, reason)
    _negative.move_to_end(url)
    while len(_negative) > MAX_NEG_ENTRIES:
        _negative.popitem(last=False)


def _retrieve_future_exception(fut: asyncio.Future) -> None:
    # If every waiter detached (all cancelled via asyncio.shield) the set exception
    # would otherwise be logged as "never retrieved"; read it here to mark it seen.
    if not fut.cancelled():
        fut.exception()


async def get_or_fetch(url: str) -> tuple[str, str]:
    now = asyncio.get_running_loop().time()
    async with _cache_lock:
        ent = _positive.get(url)
        if ent and ent[0] > now:
            _positive.move_to_end(url)
            return ent[1], ent[2]
        neg = _negative.get(url)
        if neg and neg[0] > now:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Failed to fetch image URL {url}: {neg[1]}",
            )
        fut = _inflight.get(url)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            fut.add_done_callback(_retrieve_future_exception)
            _inflight[url] = fut
            task = asyncio.create_task(_run_fetch(url, fut))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    return await asyncio.shield(fut)


async def _run_fetch(url: str, fut: asyncio.Future) -> None:
    try:
        try:
            async with _fetch_semaphore:
                raw, mime = await _fetch_bytes(url)
            b64 = base64.b64encode(raw).decode()
        except ImageFetchError as e:
            ttl = NEG_CACHE_TTL_TRANSIENT if e.transient else NEG_CACHE_TTL_DETERMINISTIC
            async with _cache_lock:
                _cache_negative(url, ttl, e.reason)
            logger.warning("image fetch failed url=%s reason=%s transient=%s", url, e.reason, e.transient)
            if not fut.done():
                fut.set_exception(
                    HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST,
                        detail=f"Failed to fetch image URL {url}: {e.reason}",
                    )
                )
            return
        except Exception as e:  # unexpected — brief negative TTL so a persistent
            async with _cache_lock:  # bug can't drive a tight per-request refetch loop
                _cache_negative(url, NEG_CACHE_TTL_UNEXPECTED, type(e).__name__)
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
            _negative.pop(url, None)  # a fresh success clears any stale negative entry
            _cache_put(url, b64, mime)
        if not fut.done():
            fut.set_result((b64, mime))
    finally:
        # Always free the inflight slot — even on CancelledError (a BaseException the
        # excepts above don't catch) — so a cancelled fetch can't leave the URL wedged
        # with an unresolved future that future waiters would block on forever.
        async with _cache_lock:
            if _inflight.get(url) is fut:
                del _inflight[url]
        if not fut.done():
            fut.cancel()


async def inline_remote_images(body_json: dict) -> tuple[dict, bool]:
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
