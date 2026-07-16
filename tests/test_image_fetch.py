import asyncio
import base64
import ipaddress

import httpx
import pytest
from fastapi import HTTPException

import src.image_fetch as imgf
from src.image_fetch import (
    _ImagePart,
    _collect_image_parts,
    _is_public_ip,
    _rewrite,
    _sniff_mime,
    inline_remote_images,
)


def _ImagePartFor(part, kind):
    return _ImagePart(part=part, kind=kind, url=part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else "")


@pytest.mark.parametrize(
    "addr, expected",
    [
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.0.0.5", False),
        ("172.16.3.4", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),   # cloud metadata
        ("100.64.0.1", False),         # CGNAT
        ("0.0.0.0", False),
        ("::1", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("::ffff:127.0.0.1", False),   # IPv4-mapped bypass on Python 3.11.2
    ],
)
def test_is_public_ip(addr, expected):
    assert _is_public_ip(ipaddress.ip_address(addr)) is expected


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8
BMP = b"BM" + b"\x00" * 16


def test_sniff_mime():
    assert _sniff_mime(PNG) == "image/png"
    assert _sniff_mime(JPEG) == "image/jpeg"
    assert _sniff_mime(GIF) == "image/gif"
    assert _sniff_mime(WEBP) == "image/webp"
    assert _sniff_mime(BMP) is None          # BMP deliberately rejected
    assert _sniff_mime(b"not an image") is None
    assert _sniff_mime(b"") is None


URL = "https://example.com/a.png"


def test_collect_openai_chat():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": URL}},
    ]}]}
    parts = _collect_image_parts(body)
    assert len(parts) == 1
    assert parts[0].kind == "openai_chat"
    assert parts[0].url == URL


def test_collect_openai_responses():
    body = {"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": URL},
    ]}]}
    parts = _collect_image_parts(body)
    assert [(p.kind, p.url) for p in parts] == [("openai_responses", URL)]


def test_collect_anthropic_url_source():
    body = {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": URL}},
    ]}]}
    parts = _collect_image_parts(body)
    assert [(p.kind, p.url) for p in parts] == [("anthropic", URL)]


def test_collect_ignores_data_urls_and_base64_sources():
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
    ]}]}
    assert _collect_image_parts(body) == []


def test_collect_repeated_url_yields_each_occurrence():
    body = {"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": URL}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": URL}}]},
    ]}
    assert len(_collect_image_parts(body)) == 2


def test_rewrite_each_format():
    chat = {"type": "image_url", "image_url": {"url": URL}}
    resp = {"type": "input_image", "image_url": URL}
    anth = {"type": "image", "source": {"type": "url", "url": URL}}
    _rewrite(_ImagePartFor(chat, "openai_chat"), "QUJD", "image/png")
    _rewrite(_ImagePartFor(resp, "openai_responses"), "QUJD", "image/jpeg")
    _rewrite(_ImagePartFor(anth, "anthropic"), "QUJD", "image/gif")
    assert chat["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert resp["image_url"] == "data:image/jpeg;base64,QUJD"
    assert anth["source"] == {"type": "base64", "media_type": "image/gif", "data": "QUJD"}


def _ok_resolver(ip):
    async def _r(host):
        return [ip]
    return _r


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def test_fetch_rejects_non_public_dns(monkeypatch):
    async def fake_getaddrinfo(host, *a, **k):
        return [(2, 1, 6, "", ("127.0.0.1", 0))]
    # Patch getaddrinfo on the running loop — the same loop _resolve_public_ips
    # reaches via asyncio.get_running_loop() — so the real IP filter runs.
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("http://evil.test/a.png")
    assert not ei.value.transient


async def test_fetch_rejects_bad_scheme():
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("ftp://example.com/a.png")
    assert "scheme" in ei.value.reason


async def test_fetch_happy_path(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == imgf.USER_AGENT
        assert request.headers["host"].startswith("example.com")
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    raw, mime = await imgf._fetch_bytes("https://example.com/a.png")
    assert mime == "image/png"
    assert raw == PNG


async def test_fetch_size_cap(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    big = PNG + b"\x00" * (imgf.MAX_IMAGE_BYTES + 10)

    def handler(request):
        return httpx.Response(200, content=big)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("https://example.com/big.png")
    assert "exceeds" in ei.value.reason


async def test_fetch_redirect_to_private_blocked(monkeypatch):
    # first host public, redirect target resolves private
    async def fake_resolve(host):
        if host == "example.com":
            return ["93.184.216.34"]
        raise imgf.ImageFetchError(f"resolves to non-public address for {host}", transient=False)
    monkeypatch.setattr(imgf, "_resolve_public_ips", fake_resolve)

    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    with pytest.raises(imgf.ImageFetchError):
        await imgf._fetch_bytes("https://example.com/a.png")


async def test_fetch_too_many_redirects(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))

    def handler(request):
        return httpx.Response(302, headers={"location": "https://example.com/next"})

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("https://example.com/a.png")
    assert "redirect" in ei.value.reason.lower()


async def test_fetch_redirect_to_data_uri_rejected(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))

    def handler(request):
        return httpx.Response(302, headers={"location": "data:text/html,x"})

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    # httpx itself raises InvalidURL while eagerly building the redirect request
    # (even with follow_redirects=False), before our own scheme check ever runs.
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("https://example.com/a.png")
    assert not ei.value.transient


async def test_resolve_public_ips_rejects_if_any_address_private(monkeypatch):
    import asyncio

    async def fake_getaddrinfo(host, *a, **k):
        return [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.0.0.5", 0)),
        ]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._resolve_public_ips("mixed.test")
    assert not ei.value.transient


async def test_fetch_connect_error_wrapped_transient(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))

    def handler(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("https://example.com/a.png")
    assert ei.value.transient is True


B64 = base64.b64encode(PNG).decode()


async def test_get_or_fetch_caches(monkeypatch):
    imgf._reset_cache_for_tests()
    calls = {"n": 0}

    async def fake_fetch(url):
        calls["n"] += 1
        return PNG, "image/png"

    monkeypatch.setattr(imgf, "_fetch_bytes", fake_fetch)
    a = await imgf.get_or_fetch("https://x.test/a.png")
    b = await imgf.get_or_fetch("https://x.test/a.png")
    assert a == (B64, "image/png")
    assert b == a
    assert calls["n"] == 1  # second served from cache


async def test_get_or_fetch_dedups_concurrent(monkeypatch):
    imgf._reset_cache_for_tests()
    calls = {"n": 0}

    async def slow_fetch(url):
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return PNG, "image/png"

    monkeypatch.setattr(imgf, "_fetch_bytes", slow_fetch)
    results = await asyncio.gather(*[imgf.get_or_fetch("https://x.test/a.png") for _ in range(5)])
    assert all(r == (B64, "image/png") for r in results)
    assert calls["n"] == 1  # coalesced


async def test_get_or_fetch_negative_cache(monkeypatch):
    imgf._reset_cache_for_tests()
    calls = {"n": 0}

    async def failing(url):
        calls["n"] += 1
        raise imgf.ImageFetchError("HTTP 404", transient=False)

    monkeypatch.setattr(imgf, "_fetch_bytes", failing)
    with pytest.raises(HTTPException) as e1:
        await imgf.get_or_fetch("https://x.test/missing.png")
    assert e1.value.status_code == 400
    with pytest.raises(HTTPException):
        await imgf.get_or_fetch("https://x.test/missing.png")
    assert calls["n"] == 1  # second served from negative cache


async def test_failure_reaches_all_waiters(monkeypatch):
    imgf._reset_cache_for_tests()

    async def failing(url):
        await asyncio.sleep(0.02)
        raise imgf.ImageFetchError("boom", transient=True)

    monkeypatch.setattr(imgf, "_fetch_bytes", failing)
    results = await asyncio.gather(
        *[imgf.get_or_fetch("https://x.test/f.png") for _ in range(4)],
        return_exceptions=True,
    )
    assert all(isinstance(r, HTTPException) for r in results)
    assert "https://x.test/f.png" not in imgf._inflight  # future cleaned up


async def test_cancelling_one_waiter_does_not_affect_others(monkeypatch):
    imgf._reset_cache_for_tests()

    async def slow(url):
        await asyncio.sleep(0.05)
        return PNG, "image/png"

    monkeypatch.setattr(imgf, "_fetch_bytes", slow)
    t1 = asyncio.create_task(imgf.get_or_fetch("https://x.test/a.png"))
    t2 = asyncio.create_task(imgf.get_or_fetch("https://x.test/a.png"))
    t3 = asyncio.create_task(imgf.get_or_fetch("https://x.test/a.png"))
    await asyncio.sleep(0.01)   # let all three register as waiters
    t1.cancel()
    r2, r3 = await asyncio.gather(t2, t3)
    assert r2 == (B64, "image/png")
    assert r3 == (B64, "image/png")
    with pytest.raises(asyncio.CancelledError):
        await t1
    assert imgf._positive.get("https://x.test/a.png") is not None  # populated despite cancellation


async def test_cache_put_reinsert_does_not_leak_bytes(monkeypatch):
    imgf._reset_cache_for_tests()
    imgf._cache_put("https://x.test/a.png", "A" * 100, "image/png")
    first = imgf._positive_bytes
    imgf._cache_put("https://x.test/a.png", "B" * 100, "image/png")
    assert imgf._positive_bytes == first  # not doubled


async def test_inline_no_images_is_noop():
    body = {"messages": [{"role": "user", "content": "hello"}]}
    out, changed = await inline_remote_images(body)
    assert changed is False
    assert out is body


async def test_inline_rewrites_all_occurrences(monkeypatch):
    imgf._reset_cache_for_tests()

    async def fake(url):
        return B64, "image/png"

    monkeypatch.setattr(imgf, "get_or_fetch", fake)
    body = {"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": URL}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": URL}}]},
    ]}
    out, changed = await inline_remote_images(body)
    assert changed is True
    urls = [m["content"][0]["image_url"]["url"] for m in out["messages"]]
    assert urls == [f"data:image/png;base64,{B64}"] * 2


async def test_inline_enforces_cap(monkeypatch):
    async def fake(url):
        return B64, "image/png"

    monkeypatch.setattr(imgf, "get_or_fetch", fake)
    content = [
        {"type": "image_url", "image_url": {"url": f"https://x.test/{i}.png"}}
        for i in range(5)
    ]
    body = {"messages": [{"role": "user", "content": content}]}
    with pytest.raises(HTTPException) as e:
        await inline_remote_images(body)
    assert e.value.status_code == 400


async def test_inline_propagates_fetch_400(monkeypatch):
    async def failing(url):
        raise HTTPException(status_code=400, detail="Failed to fetch image URL")

    monkeypatch.setattr(imgf, "get_or_fetch", failing)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": URL}},
    ]}]}
    with pytest.raises(HTTPException) as e:
        await inline_remote_images(body)
    assert e.value.status_code == 400


async def test_inline_fails_fast_on_first_failure(monkeypatch):
    imgf._reset_cache_for_tests()

    async def fake(url):
        if "slow" in url:
            await asyncio.sleep(5)
            return B64, "image/png"
        await asyncio.sleep(0.01)
        raise HTTPException(status_code=400, detail="Failed to fetch image URL")

    monkeypatch.setattr(imgf, "get_or_fetch", fake)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://x.test/slow.png"}},
        {"type": "image_url", "image_url": {"url": "https://x.test/bad.png"}},
    ]}]}
    with pytest.raises(HTTPException) as e:
        await asyncio.wait_for(inline_remote_images(body), timeout=1.0)
    assert e.value.status_code == 400


async def test_negative_cache_ttl_differs_by_transience(monkeypatch):
    imgf._reset_cache_for_tests()
    loop = asyncio.get_running_loop()

    async def transient_fail(url):
        raise imgf.ImageFetchError("timed out", transient=True)

    monkeypatch.setattr(imgf, "_fetch_bytes", transient_fail)
    with pytest.raises(HTTPException):
        await imgf.get_or_fetch("https://x.test/t.png")
    exp_t = imgf._negative["https://x.test/t.png"][0] - loop.time()
    assert exp_t <= imgf.NEG_CACHE_TTL_TRANSIENT + 1  # ~5s

    async def deterministic_fail(url):
        raise imgf.ImageFetchError("HTTP 404", transient=False)

    monkeypatch.setattr(imgf, "_fetch_bytes", deterministic_fail)
    with pytest.raises(HTTPException):
        await imgf.get_or_fetch("https://x.test/d.png")
    exp_d = imgf._negative["https://x.test/d.png"][0] - loop.time()
    assert exp_d > imgf.NEG_CACHE_TTL_TRANSIENT + 1  # ~60s, clearly longer


async def test_success_clears_stale_negative_entry(monkeypatch):
    imgf._reset_cache_for_tests()
    url = "https://x.test/a.png"
    imgf._negative[url] = (asyncio.get_running_loop().time() - 1, "expired")  # stale

    async def ok(u):
        return PNG, "image/png"

    monkeypatch.setattr(imgf, "_fetch_bytes", ok)
    assert await imgf.get_or_fetch(url) == (B64, "image/png")
    assert url not in imgf._negative


async def test_negative_cache_is_bounded(monkeypatch):
    imgf._reset_cache_for_tests()

    async def fail(url):
        raise imgf.ImageFetchError("HTTP 404", transient=False)  # 60s TTL, none expire

    monkeypatch.setattr(imgf, "_fetch_bytes", fail)
    for i in range(imgf.MAX_NEG_ENTRIES + 50):
        with pytest.raises(HTTPException):
            await imgf.get_or_fetch(f"https://x.test/{i}.png")
    assert len(imgf._negative) <= imgf.MAX_NEG_ENTRIES


async def test_unexpected_error_short_negative_cache(monkeypatch):
    imgf._reset_cache_for_tests()
    calls = {"n": 0}

    async def boom(url):
        calls["n"] += 1
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(imgf, "_fetch_bytes", boom)
    url = "https://x.test/bug.png"
    with pytest.raises(HTTPException) as e1:
        await imgf.get_or_fetch(url)
    assert e1.value.status_code == 400
    # second call within the (short) TTL is served from the negative cache, not refetched
    with pytest.raises(HTTPException):
        await imgf.get_or_fetch(url)
    assert calls["n"] == 1
    exp = imgf._negative[url][0] - asyncio.get_running_loop().time()
    assert exp <= imgf.NEG_CACHE_TTL_UNEXPECTED + 0.5  # ~1s, brief


async def test_fetch_redirect_chain_succeeds(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    seen = {"n": 0}

    def handler(request):
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(302, headers={"location": "https://example.com/final.png"})
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    raw, mime = await imgf._fetch_bytes("https://example.com/a.png")
    assert (raw, mime) == (PNG, "image/png")


async def test_fetch_relative_redirect(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    seen = {"n": 0}

    def handler(request):
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(302, headers={"location": "/final.png"})  # relative
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    raw, mime = await imgf._fetch_bytes("https://example.com/dir/a.png")
    assert mime == "image/png"


async def test_fetch_ipv6_host_header_bracketed_with_port(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("2606:4700::1111"))
    cap = {}

    def handler(request):
        cap["host"] = request.headers.get("host")
        cap["url"] = str(request.url)
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    await imgf._fetch_bytes("https://[2606:4700::1111]:8443/a.png")
    assert cap["host"] == "[2606:4700::1111]:8443"
    assert "[2606:4700::1111]:8443" in cap["url"]


async def test_fetch_nondefault_port_in_host_header(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    cap = {}

    def handler(request):
        cap["host"] = request.headers.get("host")
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(handler))
    await imgf._fetch_bytes("http://example.com:8080/a.png")
    assert cap["host"] == "example.com:8080"


async def test_fetch_malformed_port_is_deterministic():
    with pytest.raises(imgf.ImageFetchError) as ei:
        await imgf._fetch_bytes("https://example.com:99999/a.png")
    assert not ei.value.transient  # not the unexpected-error bucket


async def test_fetch_gif_jpeg_webp(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    for content, expected in [(GIF, "image/gif"), (JPEG, "image/jpeg"), (WEBP, "image/webp")]:
        monkeypatch.setattr(imgf, "_fetch_client", _mock_client(lambda r, c=content: httpx.Response(200, content=c)))
        _, mime = await imgf._fetch_bytes("https://example.com/x")
        assert mime == expected


async def test_fetch_exactly_max_bytes_ok(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))
    content = PNG + b"\x00" * (imgf.MAX_IMAGE_BYTES - len(PNG))  # exactly the cap
    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(lambda r: httpx.Response(200, content=content)))
    raw, mime = await imgf._fetch_bytes("https://example.com/x")
    assert len(raw) == imgf.MAX_IMAGE_BYTES and mime == "image/png"


async def test_positive_cache_lru_eviction_by_bytes(monkeypatch):
    imgf._reset_cache_for_tests()
    monkeypatch.setattr(imgf, "CACHE_MAX_BYTES", 300)
    imgf._cache_put("u1", "A" * 200, "image/png")
    imgf._cache_put("u2", "B" * 200, "image/png")  # 400 > 300 → oldest (u1) evicted
    assert "u1" not in imgf._positive
    assert "u2" in imgf._positive
    assert imgf._positive_bytes == 200


async def test_positive_cache_ttl_expiry_refetches(monkeypatch):
    imgf._reset_cache_for_tests()
    calls = {"n": 0}

    async def fetch(url):
        calls["n"] += 1
        return PNG, "image/png"

    monkeypatch.setattr(imgf, "_fetch_bytes", fetch)
    await imgf.get_or_fetch("u")
    _, b64, mime = imgf._positive["u"]
    imgf._positive["u"] = (asyncio.get_running_loop().time() - 1, b64, mime)  # force expired
    await imgf.get_or_fetch("u")
    assert calls["n"] == 2


async def test_inline_mixes_data_and_remote(monkeypatch):
    imgf._reset_cache_for_tests()

    async def fake(url):
        return B64, "image/png"

    monkeypatch.setattr(imgf, "get_or_fetch", fake)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZ"}},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]}]}
    out, changed = await inline_remote_images(body)
    assert changed
    c = out["messages"][0]["content"]
    assert c[0]["image_url"]["url"] == "data:image/png;base64,ZZ"       # data: untouched
    assert c[1]["image_url"]["url"] == f"data:image/png;base64,{B64}"   # remote inlined


async def test_inline_exactly_max_images_ok(monkeypatch):
    imgf._reset_cache_for_tests()

    async def fake(url):
        return B64, "image/png"

    monkeypatch.setattr(imgf, "get_or_fetch", fake)
    content = [{"type": "image_url", "image_url": {"url": f"https://x.test/{i}.png"}}
               for i in range(imgf.MAX_IMAGES_PER_REQUEST)]  # exactly the cap
    body = {"messages": [{"role": "user", "content": content}]}
    out, changed = await inline_remote_images(body)
    assert changed
    assert all(p["image_url"]["url"].startswith("data:") for p in out["messages"][0]["content"])


async def test_cancelled_fetch_frees_inflight(monkeypatch):
    imgf._reset_cache_for_tests()
    started = asyncio.Event()

    async def hang(url):
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(imgf, "_fetch_bytes", hang)
    waiter = asyncio.create_task(imgf.get_or_fetch("https://x.test/hang.png"))
    await started.wait()
    for bt in list(imgf._background_tasks):
        bt.cancel()
    waiter.cancel()
    await asyncio.gather(waiter, *list(imgf._background_tasks), return_exceptions=True)
    assert "https://x.test/hang.png" not in imgf._inflight  # freed by the finally


async def test_cache_put_purges_expired_entries():
    imgf._reset_cache_for_tests()
    imgf._cache_put("old", "A" * 100, "image/png")
    # force it expired, then insert a new entry — the stale one and its bytes go away
    _, b64, mime = imgf._positive["old"]
    imgf._positive["old"] = (asyncio.get_running_loop().time() - 1, b64, mime)
    imgf._cache_put("new", "B" * 50, "image/png")
    assert "old" not in imgf._positive
    assert imgf._positive_bytes == 50


async def test_fetch_429_is_transient_404_is_not(monkeypatch):
    monkeypatch.setattr(imgf, "_resolve_public_ips", _ok_resolver("93.184.216.34"))

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(lambda r: httpx.Response(429)))
    with pytest.raises(imgf.ImageFetchError) as rate_limited:
        await imgf._fetch_bytes("https://example.com/x")
    assert rate_limited.value.transient  # 429 = retry later (the wikimedia case)

    monkeypatch.setattr(imgf, "_fetch_client", _mock_client(lambda r: httpx.Response(404)))
    with pytest.raises(imgf.ImageFetchError) as not_found:
        await imgf._fetch_bytes("https://example.com/x")
    assert not not_found.value.transient  # 404 = deterministic
