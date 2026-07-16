import ipaddress

import pytest

from src.image_fetch import _is_public_ip, _sniff_mime, _ImagePart


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


from src.image_fetch import _collect_image_parts, _rewrite

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


import httpx

import src.image_fetch as imgf


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def test_fetch_rejects_non_public_dns(monkeypatch):
    async def fake_getaddrinfo(host, *a, **k):
        return [(2, 1, 6, "", ("127.0.0.1", 0))]
    monkeypatch.setattr(imgf.asyncio.get_event_loop(), "getaddrinfo", fake_getaddrinfo, raising=False)
    monkeypatch.setattr(imgf, "_resolve_public_ips", imgf._resolve_public_ips)  # keep real
    # patch getaddrinfo on the running loop used inside _resolve_public_ips
    import asyncio
    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo, raising=False)
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
    def resolver(host):
        async def _r(h=host):
            return ["93.184.216.34"] if h == "example.com" else None
        return _r

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

    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo, raising=False)
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
