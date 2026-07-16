import ipaddress

import pytest

from src.image_fetch import _is_public_ip, _sniff_mime


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
