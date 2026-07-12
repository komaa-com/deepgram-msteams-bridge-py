import pytest

from deepgram_msteams_bridge.ssrf import assert_public_http_url, is_forbidden_ip


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # CGNAT
        "0.0.0.0",
        "224.0.0.1",  # multicast
        "255.255.255.255",
        "198.18.0.1",  # benchmarking
        "192.0.0.1",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "fd12:3456::1",
        "::ffff:10.0.0.1",  # v4-mapped RFC1918
        "::ffff:169.254.169.254",
        "64:ff9b::a00:1",  # NAT64-embedded 10.0.0.1
        "not-an-ip",
    ],
)
def test_forbidden_ips(ip):
    assert is_forbidden_ip(ip)


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700::1111", "::ffff:8.8.8.8"])
def test_public_ips_allowed(ip):
    assert not is_forbidden_ip(ip)


async def _lookup_public(host):
    return ["93.184.216.34"]


async def _lookup_private(host):
    return ["93.184.216.34", "10.0.0.5"]


async def test_url_validation_accepts_public():
    url = await assert_public_http_url("https://example.com/cat.jpg", _lookup_public)
    assert url.startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "https://user:pass@example.com/x",
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://169.254.169.254/latest/meta-data/",
    ],
)
async def test_url_validation_rejects(url):
    with pytest.raises(ValueError):
        await assert_public_http_url(url, _lookup_public)


async def test_url_validation_rejects_private_resolution():
    with pytest.raises(ValueError, match="private/reserved"):
        await assert_public_http_url("https://rebind.example.com/x", _lookup_private)
