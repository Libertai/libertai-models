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
