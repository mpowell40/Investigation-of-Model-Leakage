"""Minimal pcap/pcapng reader: timestamps, sizes, IP addresses and ports only.

Standard library only, so the checks run on any machine with Python and no
tshark. It never looks at payloads: metadata (size, direction, timing) only.
"""
import ipaddress
import struct
from collections import namedtuple

Packet = namedtuple("Packet", "ts length proto src sport dst dport")

ETH, RAW, RAW_ALT, SLL, SLL2, LINUX_NULL = 1, 101, 12, 113, 276, 0


def _tsres(value):
    return 2.0 ** -(value & 0x7F) if value & 0x80 else 10.0 ** -value


def _ip(data, off, ethertype):
    """Decode IP + TCP/UDP header starting at data[off]. Returns (proto, src, sport, dst, dport) or None."""
    try:
        if ethertype == 0x0800:
            ihl = (data[off] & 0x0F) * 4
            p = data[off + 9]
            src = ipaddress.IPv4Address(data[off + 12:off + 16])
            dst = ipaddress.IPv4Address(data[off + 16:off + 20])
            t = off + ihl
        elif ethertype == 0x86DD:
            p = data[off + 6]
            src = ipaddress.IPv6Address(data[off + 8:off + 24])
            dst = ipaddress.IPv6Address(data[off + 24:off + 40])
            t = off + 40
        else:
            return None
        if p not in (6, 17):
            return None
        sport, dport = struct.unpack_from("!HH", data, t)
        return ("TCP" if p == 6 else "UDP", str(src), sport, str(dst), dport)
    except (IndexError, struct.error, ValueError):
        return None


def _decode(linktype, data):
    if linktype == ETH:
        if len(data) < 14:
            return None
        et, off = struct.unpack_from("!H", data, 12)[0], 14
        while et in (0x8100, 0x88A8) and len(data) >= off + 4:   # VLAN tags
            et, off = struct.unpack_from("!H", data, off + 2)[0], off + 4
        return _ip(data, off, et)
    if linktype in (RAW, RAW_ALT):
        if not data:
            return None
        return _ip(data, 0, 0x0800 if data[0] >> 4 == 4 else 0x86DD)
    if linktype == SLL and len(data) >= 16:
        return _ip(data, 16, struct.unpack_from("!H", data, 14)[0])
    if linktype == SLL2 and len(data) >= 20:
        return _ip(data, 20, struct.unpack_from("!H", data, 0)[0])
    if linktype == LINUX_NULL and len(data) >= 4:
        fam = struct.unpack_from("<I", data, 0)[0]
        return _ip(data, 4, 0x0800 if fam == 2 else 0x86DD)
    return None


def _packet(ts, origlen, linktype, data):
    d = _decode(linktype, data)
    if d is None:
        return Packet(ts, origlen, None, None, None, None, None)
    return Packet(ts, origlen, *d)


def _iter_pcap(buf):
    magic = buf[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        e = "<"
    else:
        e = ">"
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    linktype = struct.unpack_from(e + "I", buf, 20)[0] & 0x0FFFFFFF
    off = 24
    while off + 16 <= len(buf):
        sec, frac, incl, orig = struct.unpack_from(e + "IIII", buf, off)
        off += 16
        data = buf[off:off + incl]
        off += incl
        yield _packet(sec + frac * (1e-9 if nano else 1e-6), orig, linktype, data)


def _iter_pcapng(buf):
    off, e, ifaces = 0, "<", []
    while off + 12 <= len(buf):
        btype = struct.unpack_from("<I", buf, off)[0]
        if btype == 0x0A0D0D0A:                      # section header: fixes byte order
            e = "<" if buf[off + 8:off + 12] == b"\x4d\x3c\x2b\x1a" else ">"
            ifaces = []
        btype, blen = struct.unpack_from(e + "II", buf, off)
        if blen < 12 or off + blen > len(buf):
            break                                    # truncated file: stop cleanly
        body = buf[off + 8:off + blen - 4]
        if btype == 1:                               # interface description
            linktype = struct.unpack_from(e + "H", body, 0)[0]
            res = 1e-6
            o = 8
            while o + 4 <= len(body):
                code, olen = struct.unpack_from(e + "HH", body, o)
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    res = _tsres(body[o + 4])
                o += 4 + ((olen + 3) & ~3)
            ifaces.append((linktype, res))
        elif btype == 6 and len(body) >= 20:         # enhanced packet
            iid, hi, lo, cap, orig = struct.unpack_from(e + "IIIII", body, 0)
            linktype, res = ifaces[iid] if iid < len(ifaces) else (ETH, 1e-6)
            yield _packet(((hi << 32) | lo) * res, orig, linktype, body[20:20 + cap])
        off += blen


def read_packets(path):
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 24:
        return []
    if buf[:4] == b"\x0a\x0d\x0d\x0a":
        return list(_iter_pcapng(buf))
    return list(_iter_pcap(buf))


def capture_stats(path):
    """(packet count, duration in seconds) for quick sanity checks."""
    pkts = read_packets(path)
    if len(pkts) < 2:
        return len(pkts), 0.0
    return len(pkts), pkts[-1].ts - pkts[0].ts
