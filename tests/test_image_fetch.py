import ipaddress

import pytest

from src.image_fetch import _is_public_ip


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
