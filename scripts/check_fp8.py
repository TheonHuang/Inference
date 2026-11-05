#!/usr/bin/env python3
"""Local numeric validation for include/inferop/fp8.h (no compiler needed).

Transcribes the exact integer algorithms from fp8.h and checks them against an
independent oracle that enumerates the 256-value E4M3 grid from the OCP spec.
The same oracle logic is compiled into tests/test_fp8.cpp for CI.
"""
import math
import random
import struct
import sys


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def e4m3_to_float(b):  # transcription of fp8::e4m3_to_float
    s = (b >> 7) << 31
    E = (b >> 3) & 0xF
    m = b & 7
    if E == 15 and m == 7:
        return float("nan")
    if E == 0:
        if m == 0:
            return struct.unpack("<f", struct.pack("<I", s))[0]
        mm, e32 = m, 121
        while mm < 8:
            mm <<= 1
            e32 -= 1
        w = s | (e32 << 23) | ((mm & 7) << 20)
        return struct.unpack("<f", struct.pack("<I", w))[0]
    w = s | ((E + 120) << 23) | (m << 20)
    return struct.unpack("<f", struct.pack("<I", w))[0]


def float_to_e4m3_rn_sat(f):  # transcription of fp8::float_to_e4m3_rn_sat
    x = bits(f32(f))
    sign = (x >> 24) & 0x80
    a = x & 0x7FFFFFFF
    if a > 0x7F800000:
        return sign | 0x7F
    if a == 0x7F800000:
        return sign | 0x7E
    e = ((a >> 23) & 0xFF) - 127
    if e >= 9:
        return sign | 0x7E
    if e >= -6:
        man = (a & 0x7FFFFF) | 0x800000
        keep = man >> 20
        rem = man & 0xFFFFF
        r = 1 if (rem > 0x80000 or (rem == 0x80000 and (keep & 1))) else 0
        m = keep - 8 + r
        E = e + 7
        if m == 8:
            m, E = 0, E + 1
        if E >= 16 or (E == 15 and m == 7):
            return sign | 0x7E
        return sign | (E << 3) | m
    if e >= -10:
        man = (a & 0x7FFFFF) | 0x800000
        sh = 14 - e
        keep = man >> sh
        rem = man & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        r = 1 if (rem > half or (rem == half and (keep & 1))) else 0
        md = keep + r
        if md == 8:
            return sign | 0x08
        return sign | md
    return sign


def grid():  # independent enumeration from the OCP definition
    g = [0.0] * 256
    for b in range(256):
        sign = -1.0 if (b >> 7) else 1.0
        E, m = (b >> 3) & 0xF, b & 7
        if E == 15 and m == 7:
            v = float("nan")
        elif E == 0:
            v = 0.0 if m == 0 else m * 2.0 ** -9
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (E - 7.0)
        g[b] = sign * v
    return g


def oracle_encode(f):  # nearest grid point, ties -> even encoding
    if math.isnan(f):
        return 0x7F
    g = grid()
    neg = math.copysign(1.0, f) < 0
    a = abs(f)
    if a == 0.0:
        return 0x80 if neg else 0x00
    if a >= 448.0:  # nothing representable above 448 -> saturate (SATFINITE)
        return 0xFE if neg else 0x7E
    best, bestd = 0, float("inf")
    for b in range(0x7F):  # positive encodings, skip NaN slot
        d = abs(a - g[b])
        if d < bestd:
            best, bestd = b, d
        elif d == bestd and (b & 1) == 0 and (best & 1) == 1:
            best = b
    return (best | 0x80) if neg else best


def main():
    rng = random.Random(7)
    failures = 0

    # 1) decode exactness
    g = grid()
    for b in range(256):
        want, got = g[b], e4m3_to_float(b)
        if math.isnan(want):
            ok = math.isnan(got)
        else:
            ok = (got == want) or (want == 0.0 and got == 0.0 and math.copysign(1, want) == math.copysign(1, got))
        if not ok:
            failures += 1
            print(f"decode fail: b=0x{b:02x} got={got!r} want={want!r}")
    print(f"decode: {'OK' if failures == 0 else 'FAIL'} (256 values)")

    # 2) encode vs oracle
    samples = []
    for _ in range(200000):
        mag = 10 ** rng.uniform(-9, 3)
        samples.append(math.copysign(mag, rng.choice((-1.0, 1.0))))
    samples += [0.0, -0.0, 1e-45, -1e-45, 1e38, -1e38, float("inf"), float("-inf")]
    for b in range(1, 0x7F):
        if b == 0x7E:
            samples.append((g[b] + 512.0) / 2)
            continue
        samples.append(g[b])
        samples.append((g[b] + g[b + 1]) / 2)
    enc_fail = 0
    for f in samples:
        if math.isnan(f):
            continue
        got, want = float_to_e4m3_rn_sat(f), oracle_encode(f)
        if got != want:
            enc_fail += 1
            if enc_fail <= 5:
                print(f"encode fail: f={f!r} got=0x{got:02x} want=0x{want:02x}")
    print(f"encode vs oracle: {'OK' if enc_fail == 0 else 'FAIL'} ({len(samples)} samples, {enc_fail} bad)")

    sys.exit(0 if (failures == 0 and enc_fail == 0) else 1)


if __name__ == "__main__":
    main()
