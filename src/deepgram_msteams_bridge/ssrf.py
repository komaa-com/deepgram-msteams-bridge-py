"""SSRF guard for agent-supplied URLs (show_image).

The tool params come from the agent's LLM, i.e. indirectly from the caller - a
crafted prompt must not be able to make the bridge fetch cloud metadata
(169.254.169.254), loopback, or RFC1918 hosts. Mirrors the same guard the
StandIn media bridge applies.

The DNS-rebind TOCTOU (validate-then-fetch re-resolves, and the second answer
can be private) is CLOSED for fetch_public_image: the actual connect goes
through a guarded resolver that re-checks every address against the same
private-range rules, so a rebind to a private IP fails at connect time.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Awaitable, Callable
from urllib.parse import urlsplit, urljoin

import aiohttp
from aiohttp import abc as aiohttp_abc

# addresses that must never be fetched server-side (beyond ipaddress's own flags)
_FORBIDDEN_V4 = [
    ipaddress.ip_network("0.0.0.0/8"),  # "this" network
    ipaddress.ip_network("10.0.0.0/8"),  # RFC1918
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("169.254.0.0/16"),  # link-local incl. cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),  # RFC1918
    ipaddress.ip_network("192.0.0.0/24"),  # IETF protocol assignments
    ipaddress.ip_network("192.168.0.0/16"),  # RFC1918
    ipaddress.ip_network("198.18.0.0/15"),  # benchmarking
    ipaddress.ip_network("224.0.0.0/3"),  # multicast + reserved + broadcast
]

LookupFn = Callable[[str], Awaitable[list[str]]]


def _is_forbidden_v4(addr: ipaddress.IPv4Address) -> bool:
    return any(addr in net for net in _FORBIDDEN_V4)


def _is_forbidden_v6(addr: ipaddress.IPv6Address) -> bool:
    # v4-mapped/translated (::ffff:a.b.c.d, 64:ff9b::/96) -> judge the embedded v4
    v4 = addr.ipv4_mapped
    if v4 is not None:
        return _is_forbidden_v4(v4)
    if addr in ipaddress.ip_network("64:ff9b::/96"):
        return _is_forbidden_v4(ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF))
    if addr == ipaddress.IPv6Address("::") or addr.is_loopback:
        return True  # unspecified / loopback
    if addr in ipaddress.ip_network("fc00::/7"):
        return True  # unique-local
    if addr.is_link_local:
        return True  # fe80::/10
    return False


def is_forbidden_ip(ip: str) -> bool:
    """True for any address that must never be fetched server-side (or junk input)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if isinstance(addr, ipaddress.IPv4Address):
        return _is_forbidden_v4(addr)
    return _is_forbidden_v6(addr)


async def _default_lookup(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def assert_public_http_url(raw: str, lookup: LookupFn = _default_lookup) -> str:
    """Validate an outbound URL: http(s) only, no credentials, and every address
    the host resolves to must be public. Raises ValueError with a reason."""
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"forbidden protocol {parts.scheme or '(none)'}:")
    if parts.username or parts.password:
        raise ValueError("URLs with embedded credentials are not allowed")
    host = parts.hostname
    if not host:
        raise ValueError("not a valid URL")
    try:
        ipaddress.ip_address(host)
        is_literal = True
    except ValueError:
        is_literal = False
    if is_literal:
        if is_forbidden_ip(host):
            raise ValueError(f"address {host} is private/reserved")
        return raw
    try:
        addrs = await lookup(host)
    except Exception:
        raise ValueError(f"cannot resolve host {host}") from None
    if not addrs:
        raise ValueError(f"host {host} resolves to no addresses")
    for a in addrs:
        if is_forbidden_ip(a):
            raise ValueError(f"host {host} resolves to private/reserved address {a}")
    return raw


class _GuardedResolver(aiohttp_abc.AbstractResolver):
    """Connect-time guard: aiohttp calls THIS to resolve the host, so a DNS
    rebind (public answer for the validation, private for the fetch) is rejected
    here instead of silently connecting to an internal address."""

    def __init__(self, lookup: LookupFn) -> None:
        self._lookup = lookup

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list[dict]:
        addrs = await self._lookup(host)
        bad = next((a for a in addrs if is_forbidden_ip(a)), None)
        if bad or not addrs:
            raise OSError(f"DNS rebind blocked: {host} resolved to {bad or 'nothing'}")
        results = []
        for a in addrs:
            fam = socket.AF_INET6 if ":" in a else socket.AF_INET
            # honor the family aiohttp asked for (AF_UNSPEC = both): returning
            # the wrong family breaks connection setup on dual-stack hosts
            if family not in (socket.AF_UNSPEC, fam):
                continue
            results.append(
                {
                    "hostname": host,
                    "host": a,
                    "port": port,
                    "family": fam,
                    "proto": 0,
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        if not results:
            raise OSError(f"{host} has no addresses for the requested address family")
        return results

    async def close(self) -> None:  # pragma: no cover - nothing to release
        return None


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


async def fetch_public_image(
    raw_url: str,
    max_bytes: int,
    timeout_ms: float = 10_000,
    lookup: LookupFn = _default_lookup,
    redirects_left: int = 1,
) -> tuple[bytes, str]:
    """Fetch an image from an untrusted (agent/LLM-supplied) URL with the full
    SSRF posture: public http(s) host, no credentials, connect-time DNS guarded
    against rebind, bounded total time and body size. At most ONE redirect hop
    is followed, and the redirect target goes through the exact same validation
    as the original URL (image CDNs commonly 301/302 to the actual asset; an
    unvalidated redirect would be a guard bypass). Returns (bytes, mime)."""
    url = await assert_public_http_url(raw_url, lookup)
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    connector = aiohttp.TCPConnector(resolver=_GuardedResolver(lookup))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(url, allow_redirects=False, headers={"accept": "image/*"}) as res:
            # One re-validated redirect hop: the Location target is resolved
            # against the current URL and recursed through the FULL guard.
            if res.status in _REDIRECT_STATUSES:
                location = res.headers.get("location")
                if not location:
                    raise ValueError(f"fetch {raw_url} -> HTTP {res.status} with no Location header")
                if redirects_left <= 0:
                    raise ValueError(f"fetch {raw_url} -> too many redirects")
                next_url = urljoin(url, location)
                return await fetch_public_image(next_url, max_bytes, timeout_ms, lookup, redirects_left - 1)
            if res.status != 200:
                raise ValueError(f"fetch {raw_url} -> HTTP {res.status}")
            declared = res.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ValueError(f"response too large ({declared} bytes, max {max_bytes})")
            mime = (res.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
            chunks: list[bytes] = []
            total = 0
            async for chunk in res.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"response exceeded {max_bytes} bytes; aborting read")
                chunks.append(chunk)
            return b"".join(chunks), mime
