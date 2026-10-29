#!/usr/bin/env python3
"""Write the container::metadata coverage seeds to fuzz/seeds/metadata/.

Each seed is a small synthetic JPEG (< 1 KB) that reaches one group of
`container::metadata` paths: every identified APP format, the vendor trailer
chains, and the resynchronization paths. `tests/fuzz_regression.rs` replays
them through every entry point, including `fuzz_container_metadata`'s checks
(`fuzz/metadata_checks.rs`), and the ExifTool oracle test checks them against
ExifTool. Layouts follow the readers cited in `zenjpeg/docs/METADATA.md`.

Usage: python3 fuzz/make_metadata_seeds.py   (from zenjpeg/)
"""
import hashlib
import os
import struct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds", "metadata")


def seg(marker, payload):
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


SOI, EOI = b"\xff\xd8", b"\xff\xd9"
SOF = seg(0xC0, bytes([8, 0, 8, 0, 16, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]))
SOS = seg(0xDA, bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0])) + bytes(
    [0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56]
)


def jpeg(*segments):
    return SOI + b"".join(segments) + SOF + SOS + EOI


def tiff(tag, value):
    t = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1)
    t += struct.pack("<HHI", tag, 7, len(value)) + struct.pack("<I", 8 + 2 + 12 + 4)
    return t + struct.pack("<I", 0) + value


def ducky_record(tag, data):
    return struct.pack(">HH", tag, len(data)) + data


def ducky_text(text):
    """A Ducky text value: a 4-byte character count, then UTF-16BE (APP12.pm
    skips the count)."""
    return struct.pack(">I", len(text)) + text.encode("utf-16-be")


def fotostation(tag, data):
    return data + struct.pack(">HI", tag, len(data) + 10) + b"\xa1\xb2\xc3\xd4"


ext = b"<x:ext/>"
guid = hashlib.md5(ext).hexdigest().upper().encode()

app_a = [
    seg(0xE0, b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x01\x01\xaa\xbb\xcc"),
    # Table segments: DQT (table 0, 64 values), DHT (one 1-bit code), DAC,
    # DNL, so the seeds reach every frame-structure kind.
    seg(0xDB, bytes([0]) + bytes(range(1, 65))),
    seg(0xC4, bytes([0x00, 1]) + bytes(15) + bytes([0x00])),
    seg(0xCC, b"\x00\x11"),
    seg(0xDC, b"\x00\x08"),
    seg(0xE0, b"JFXX\0\x10\xff\xd8\xff\xd9"),
    seg(0xE0, b"AVI1\x01\0\0\0\0\0\0\0"),
    seg(0xE0, b"II\x1a\0\0\0HEAPJPGM\0\0"),
    seg(0xE1, b"Exif\0\0" + tiff(0x0112, b"\x01\x00")),
    seg(0xE1, b"http://ns.adobe.com/xap/1.0/\0<x xmpNote:HasExtendedXMP=\"" + guid + b"\"/>"),
    seg(0xE1, b"http://ns.adobe.com/xmp/extension/\0" + guid + struct.pack(">II", len(ext), 0) + ext),
    seg(0xE2, b"ICC_PROFILE\0\x02\x02bb"),
    seg(0xE2, b"ICC_PROFILE\0\x01\x02aa"),
    seg(0xE2, b"FPXR\0\x00\x01"),
    seg(0xE2, b"MPF\0MM\0*\0\0\0\x08\0\0"),
    seg(0xE2, b"urn:iso:std:iso:ts:21496:-1\0\0\0"),
    seg(0xE4, b"SCALADO\0"),
]
app_b = [
    seg(0xE3, b"Meta\0\0II*\0"),
    seg(0xE5, b"RMETA\0II"),
    seg(0xE6, b"EPPIM\0" + tiff(0xC4A5, b"PrintIM\x000300\0\0\0\0")),
    seg(0xE8, b"SPIFF\0\x01\x00\x01\x03" + struct.pack(">II", 8, 16) + bytes([3, 8, 5, 1]) + struct.pack(">II", 72, 72)),
    seg(0xEA, b"UNICODE\0\0H\0i"),
    seg(0xEB, b"JP\x00\x01" + struct.pack(">II", 1, 24) + b"jumbcontent!"),
    # Each Ducky tag twice: ExifTool gives all three priority 0, so the first
    # value of each stays (Quality 11, Comment "first", Copyright "c1").
    seg(
        0xEC,
        b"Ducky"
        + ducky_record(1, struct.pack(">I", 11))
        + ducky_record(1, struct.pack(">I", 22))
        + ducky_record(2, ducky_text("first"))
        + ducky_record(2, ducky_text("second"))
        + ducky_record(3, ducky_text("c1"))
        + ducky_record(3, ducky_text("c2"))
        + b"\0\0",
    ),
    seg(0xEC, b"[picture info]\r\nTimeDate=966665853\r\n"),
    seg(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00" + struct.pack(">I", 3) + b"abc\0"),
    seg(0xED, b"Photoshop 3.0\x008BIM\x04\x05\x01x" + struct.pack(">I", 1) + b"z\0"),
    seg(0xED, b"Adobe_CM\x00\x02"),
    seg(0xED, b"Adobe_Photoshop2.5:" + b"\0" * 8 + b"8BIM\x04\x04\x00\x00" + struct.pack(">I", 2) + b"ok"),
    seg(0xEE, b"Adobe\x00\x64\x80\x00\x00\x00\x01"),
    seg(0xEF, b"Q 85"),
    seg(0xFE, b"comment\0"),
    seg(0xDD, b"\x00\x10"),
]

base = jpeg()
# A MIE group opens with the 0MIE element (MIE.pm ProcessMIE) and ends with
# the zmie trailer signature and a group terminator.
mie_group = b"~\x10\x04\x000MIE" + b"~\0\x04\0zmie"
mie = mie_group + b"~\0\0\x06" + struct.pack(">I", len(mie_group) + 10) + b"\x10\x04"
vrd = b"CANON OPTIONAL DATA\0".ljust(0x1C, b"\0") + b"VRD!"
vrd += (b"CANON OPTIONAL DATA\0".ljust(0x14, b"\0") + struct.pack(">I", 4)).ljust(0x40, b"\0")
blk = b"\0\0\x01\x0aImage_UTC_Data1234"
sefh = b"SEFH" + struct.pack("<II", 101, 1) + b"\0\0\x01\x0a" + struct.pack("<II", len(blk), len(blk))
samsung = blk + sefh + struct.pack("<I", len(sefh)) + b"SEFT"
afcp_body = b"AXS!" + b"\0" * 8 + b"IPTC"

def mpf_app2(primary_len, second_len, second_start):
    """APP2 MPF (CIPA DC-007): a big-endian TIFF with the version, image
    count and MP Entry tags, then two 16-byte entries. The segment sits right
    after SOI, so its TIFF header is at file offset 2 + 4 + 4 = 10, and the
    second image's offset is relative to it."""
    tiff = b"MM\x00\x2a" + struct.pack(">I", 8) + struct.pack(">H", 3)
    tiff += struct.pack(">HHI", 0xB000, 7, 4) + b"0100"
    tiff += struct.pack(">HHII", 0xB001, 4, 1, 2)
    tiff += struct.pack(">HHII", 0xB002, 7, 32, 8 + 2 + 3 * 12 + 4)
    tiff += struct.pack(">I", 0)
    tiff += struct.pack(">IIIHH", 0x20030000, primary_len, 0, 0, 0)
    tiff += struct.pack(">IIIHH", 0x00010001, second_len, second_start - 10, 0, 0)
    return seg(0xE2, b"MPF\0" + tiff)


second = jpeg(seg(0xFE, b"second"))
primary_len = len(base) + len(mpf_app2(0, 0, 10))
mpf_primary = SOI + mpf_app2(primary_len, len(second), primary_len + 8) + base[2:]
assert len(mpf_primary) == primary_len

seeds = {
    "metadata-app-formats-a": jpeg(*app_a),
    "metadata-app-formats-b": jpeg(*app_b),
    "metadata-trailer-mpf-pm-fotostation": mpf_primary
    + b"\0" * 8
    + second
    + b"\x1c\x02\x00\x00\x02\x00\x04"
    + struct.pack(">I", 7)
    + b"cbipcbbl"
    + fotostation(1, b"iptc")
    + fotostation(4, b"preview"),
    "metadata-trailer-vrd-samsung": base + vrd + samsung,
    "metadata-trailer-mie-afcp": (lambda pre: pre + afcp_body + b"AXS!" + struct.pack(">I", len(pre)) + b"\0" * 4)(base + mie),
    "metadata-resync": SOI
    + b"\x01\x02"
    + seg(0xFE, b"a")
    + b"\xff\x00\x07"
    + b"\xff\xe4\x00\x01\x09\x09"
    + b"\x11\xff\xd0\x22"
    + b"\xff\xff"
    + seg(0xFE, b"b")
    + b"\xff\xd5"
    + SOF
    + SOS
    + EOI
    + b"\xff\xd8\x55\xff\xd9"
    + b"\xff\xe1\x00\x40Exif",
    "metadata-truncated-segment": SOI + seg(0xFE, b"c") + b"\xff\xe1\x00\x40Exif\0\0",
    "metadata-icc-and-limits": jpeg(
        seg(0xE2, b"ICC_PROFILE\0\x01\x03a"),
        seg(0xE2, b"ICC_PROFILE\0\x01\x03b"),
        seg(0xE2, b"ICC_PROFILE\0\x00\x03c"),
        *[seg(0xFE, bytes([i])) for i in range(24)],
    ),
}

os.makedirs(OUT, exist_ok=True)
for name, data in seeds.items():
    assert len(data) < 1024, (name, len(data))
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)
    print(f"{name}: {len(data)} bytes")
