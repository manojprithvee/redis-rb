"""Tests for the RDB v10 parser."""

import struct
import unittest
from rdb_parser import (
    parse_rdb, RDBFile, RDBEntry,
    RDB_TYPE_STRING, RDB_TYPE_LIST, RDB_TYPE_SET, RDB_TYPE_ZSET,
    RDB_TYPE_ZSET_2, RDB_TYPE_HASH, RDB_TYPE_LIST_ZIPLIST,
    RDB_TYPE_SET_INTSET, RDB_TYPE_ZSET_ZIPLIST, RDB_TYPE_HASH_ZIPLIST,
    RDB_TYPE_LIST_QUICKLIST, RDB_TYPE_LIST_QUICKLIST_2,
    RDB_TYPE_HASH_LISTPACK, RDB_TYPE_ZSET_LISTPACK, RDB_TYPE_SET_LISTPACK,
    _decode_ziplist, _decode_intset, _decode_listpack,
    RDB_OPCODE_AUX, RDB_OPCODE_SELECTDB, RDB_OPCODE_EXPIRETIME_MS,
    RDB_OPCODE_RESIZEDB, RDB_OPCODE_EOF,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal RDB blobs in memory
# ---------------------------------------------------------------------------

def _length_encode(n: int) -> bytes:
    if n <= 63:
        return bytes([n])
    elif n <= 16383:
        return bytes([0x40 | (n >> 8), n & 0xFF])
    else:
        return b'\x80' + struct.pack(">I", n)


def _string_encode(s: bytes) -> bytes:
    return _length_encode(len(s)) + s


def _make_rdb(version: int, body: bytes, checksum: bytes = b'\x00' * 8) -> bytes:
    header = b"REDIS" + f"{version:04d}".encode()
    return header + body + bytes([RDB_OPCODE_EOF]) + checksum


def _entry(value_type: int, key: bytes, value_blob: bytes,
           expire_ms: int = None) -> bytes:
    parts = b""
    if expire_ms is not None:
        parts += bytes([RDB_OPCODE_EXPIRETIME_MS]) + struct.pack("<Q", expire_ms)
    parts += bytes([value_type]) + _string_encode(key) + value_blob
    return parts


# ---------------------------------------------------------------------------
# Ziplist builder
# ---------------------------------------------------------------------------

def _zl_encode_str(s: bytes) -> bytes:
    prevlen = b'\x00'
    if len(s) <= 63:
        enc = bytes([len(s)])
    else:
        enc = b'\x40' + bytes([len(s)])
    entry = prevlen + enc + s
    return entry


def _make_ziplist(items: list) -> bytes:
    entries = b""
    for item in items:
        b = item if isinstance(item, bytes) else str(item).encode()
        entries += _zl_encode_str(b)
    entries += b'\xFF'
    zlbytes = 4 + 4 + 2 + len(entries)
    zltail  = 4 + 4 + 2  # offset to last entry (simplified)
    zllen   = len(items)
    return (struct.pack("<I", zlbytes) +
            struct.pack("<I", zltail) +
            struct.pack("<H", zllen) +
            entries)


# ---------------------------------------------------------------------------
# Intset builder
# ---------------------------------------------------------------------------

def _make_intset(values: list, encoding: int = 8) -> bytes:
    fmt = {2: "<h", 4: "<i", 8: "<q"}[encoding]
    body = b"".join(struct.pack(fmt, v) for v in sorted(values))
    return struct.pack("<I", encoding) + struct.pack("<I", len(values)) + body


# ---------------------------------------------------------------------------
# Listpack builder (minimal, 7-bit uint only for simplicity)
# ---------------------------------------------------------------------------

def _lp_encode_uint7(val: int) -> bytes:
    assert 0 <= val <= 127
    return bytes([val & 0x7F]) + b'\x01'  # backlen = 1 (just the encoding byte)


def _lp_encode_str6(s: bytes) -> bytes:
    assert len(s) <= 63
    entry = bytes([0x80 | len(s)]) + s   # 10xxxxxx encoding
    backlen_val = 1 + len(s)             # encoding byte + string data
    assert backlen_val < 128
    return entry + bytes([backlen_val])


def _make_listpack(items: list) -> bytes:
    entries = b""
    for item in items:
        if isinstance(item, int) and 0 <= item <= 127:
            entries += _lp_encode_uint7(item)
        else:
            b = item if isinstance(item, bytes) else str(item).encode()
            entries += _lp_encode_str6(b)
    entries += b'\xFF'
    total_bytes = 4 + 2 + len(entries)
    return (struct.pack("<I", total_bytes) +
            struct.pack("<H", len(items)) +
            entries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMagicAndVersion(unittest.TestCase):

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            parse_rdb(b"WRONG0010" + bytes([RDB_OPCODE_EOF]) + b'\x00' * 8)

    def test_unsupported_version(self):
        with self.assertRaises(ValueError):
            parse_rdb(b"REDIS0011" + bytes([RDB_OPCODE_EOF]) + b'\x00' * 8)

    def test_version_10(self):
        rdb = parse_rdb(_make_rdb(10, b""))
        self.assertEqual(rdb.version, 10)

    def test_version_9(self):
        rdb = parse_rdb(_make_rdb(9, b""))
        self.assertEqual(rdb.version, 9)


class TestAuxFields(unittest.TestCase):

    def test_aux_fields_parsed(self):
        body = (bytes([RDB_OPCODE_AUX]) +
                _string_encode(b"redis-ver") + _string_encode(b"7.2.0"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.aux[b"redis-ver"], b"7.2.0")

    def test_multiple_aux(self):
        body = (bytes([RDB_OPCODE_AUX]) +
                _string_encode(b"redis-ver") + _string_encode(b"7.0") +
                bytes([RDB_OPCODE_AUX]) +
                _string_encode(b"aof-base") + _string_encode(b"0"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(len(rdb.aux), 2)


class TestSelectDB(unittest.TestCase):

    def test_selectdb(self):
        body = (bytes([RDB_OPCODE_SELECTDB]) + _length_encode(3) +
                _entry(RDB_TYPE_STRING, b"k", _string_encode(b"v")))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].db, 3)


class TestResizeDB(unittest.TestCase):

    def test_resizedb_skipped(self):
        body = (bytes([RDB_OPCODE_SELECTDB]) + _length_encode(0) +
                bytes([RDB_OPCODE_RESIZEDB]) + _length_encode(5) + _length_encode(2) +
                _entry(RDB_TYPE_STRING, b"k", _string_encode(b"v")))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(len(rdb.entries), 1)


class TestStringType(unittest.TestCase):

    def test_simple_string(self):
        body = _entry(RDB_TYPE_STRING, b"hello", _string_encode(b"world"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, b"world")
        self.assertEqual(rdb.entries[0].key, b"hello")

    def test_empty_string(self):
        body = _entry(RDB_TYPE_STRING, b"k", _string_encode(b""))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, b"")


class TestExpiry(unittest.TestCase):

    def test_expiretime_ms(self):
        body = _entry(RDB_TYPE_STRING, b"k", _string_encode(b"v"),
                      expire_ms=1_700_000_000_000)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].expire_ms, 1_700_000_000_000)

    def test_no_expiry(self):
        body = _entry(RDB_TYPE_STRING, b"k", _string_encode(b"v"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertIsNone(rdb.entries[0].expire_ms)

    def test_expiretime_seconds(self):
        from rdb_parser import RDB_OPCODE_EXPIRETIME
        ts_sec = 1_700_000_000
        body = (bytes([RDB_OPCODE_EXPIRETIME]) + struct.pack("<I", ts_sec) +
                bytes([RDB_TYPE_STRING]) + _string_encode(b"k") + _string_encode(b"v"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].expire_ms, ts_sec * 1000)


class TestListType(unittest.TestCase):

    def test_plain_list(self):
        items = [b"a", b"b", b"c"]
        blob = _length_encode(3) + b"".join(_string_encode(i) for i in items)
        body = _entry(RDB_TYPE_LIST, b"mylist", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, items)


class TestSetType(unittest.TestCase):

    def test_plain_set(self):
        items = [b"x", b"y"]
        blob = _length_encode(2) + b"".join(_string_encode(i) for i in items)
        body = _entry(RDB_TYPE_SET, b"myset", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(set(rdb.entries[0].value), {b"x", b"y"})


class TestHashType(unittest.TestCase):

    def test_plain_hash(self):
        blob = (_length_encode(2) +
                _string_encode(b"f1") + _string_encode(b"v1") +
                _string_encode(b"f2") + _string_encode(b"v2"))
        body = _entry(RDB_TYPE_HASH, b"myhash", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, {b"f1": b"v1", b"f2": b"v2"})


class TestZSetType(unittest.TestCase):

    def _score_str(self, score: float) -> bytes:
        s = str(score).encode()
        return bytes([len(s)]) + s

    def test_zset_v1(self):
        blob = (_length_encode(1) +
                _string_encode(b"member") +
                self._score_str(1.5))
        body = _entry(RDB_TYPE_ZSET, b"z", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value[0][0], b"member")
        self.assertAlmostEqual(rdb.entries[0].value[0][1], 1.5)

    def test_zset_v2_binary(self):
        blob = (_length_encode(1) +
                _string_encode(b"m") +
                struct.pack("<d", 3.14))
        body = _entry(RDB_TYPE_ZSET_2, b"z2", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertAlmostEqual(rdb.entries[0].value[0][1], 3.14)


class TestZiplistTypes(unittest.TestCase):

    def test_list_ziplist(self):
        zl = _make_ziplist([b"one", b"two", b"three"])
        body = _entry(RDB_TYPE_LIST_ZIPLIST, b"lzl", _string_encode(zl))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, [b"one", b"two", b"three"])

    def test_hash_ziplist(self):
        zl = _make_ziplist([b"field", b"value"])
        body = _entry(RDB_TYPE_HASH_ZIPLIST, b"hzl", _string_encode(zl))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, {b"field": b"value"})

    def test_zset_ziplist(self):
        zl = _make_ziplist([b"mem", b"2.5"])
        body = _entry(RDB_TYPE_ZSET_ZIPLIST, b"zzl", _string_encode(zl))
        rdb = parse_rdb(_make_rdb(10, body))
        pairs = rdb.entries[0].value
        self.assertEqual(pairs[0][0], b"mem")
        self.assertAlmostEqual(pairs[0][1], 2.5)


class TestIntset(unittest.TestCase):

    def test_intset_64bit(self):
        intset = _make_intset([1, 2, 3, 100], encoding=8)
        body = _entry(RDB_TYPE_SET_INTSET, b"is", _string_encode(intset))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(set(rdb.entries[0].value), {b"1", b"2", b"3", b"100"})

    def test_intset_16bit(self):
        intset = _make_intset([10, 20], encoding=2)
        body = _entry(RDB_TYPE_SET_INTSET, b"is2", _string_encode(intset))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(set(rdb.entries[0].value), {b"10", b"20"})


class TestListpackTypes(unittest.TestCase):

    def test_set_listpack(self):
        lp = _make_listpack([b"a", b"b"])
        body = _entry(RDB_TYPE_SET_LISTPACK, b"slp", _string_encode(lp))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(set(rdb.entries[0].value), {b"a", b"b"})

    def test_hash_listpack(self):
        lp = _make_listpack([b"k1", b"v1", b"k2", b"v2"])
        body = _entry(RDB_TYPE_HASH_LISTPACK, b"hlp", _string_encode(lp))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, {b"k1": b"v1", b"k2": b"v2"})

    def test_zset_listpack(self):
        lp = _make_listpack([b"member", b"3"])
        body = _entry(RDB_TYPE_ZSET_LISTPACK, b"zlp", _string_encode(lp))
        rdb = parse_rdb(_make_rdb(10, body))
        pairs = rdb.entries[0].value
        self.assertEqual(pairs[0][0], b"member")
        self.assertAlmostEqual(pairs[0][1], 3.0)


class TestQuicklist(unittest.TestCase):

    def test_quicklist_v1(self):
        zl = _make_ziplist([b"item1", b"item2"])
        blob = _length_encode(1) + _string_encode(zl)
        body = _entry(RDB_TYPE_LIST_QUICKLIST, b"ql", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].value, [b"item1", b"item2"])

    def test_quicklist_v2_listpack(self):
        lp = _make_listpack([b"x", b"y"])
        # container=2 means listpack; encode as length
        blob = _length_encode(1) + _string_encode(lp) + _length_encode(2)
        body = _entry(RDB_TYPE_LIST_QUICKLIST_2, b"ql2", blob)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertIn(b"x", rdb.entries[0].value)
        self.assertIn(b"y", rdb.entries[0].value)


class TestMultipleEntries(unittest.TestCase):

    def test_multiple_entries_multiple_dbs(self):
        body = (bytes([RDB_OPCODE_SELECTDB]) + _length_encode(0) +
                _entry(RDB_TYPE_STRING, b"k0", _string_encode(b"v0")) +
                bytes([RDB_OPCODE_SELECTDB]) + _length_encode(1) +
                _entry(RDB_TYPE_STRING, b"k1", _string_encode(b"v1")))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(len(rdb.entries), 2)
        self.assertEqual(rdb.entries[0].db, 0)
        self.assertEqual(rdb.entries[1].db, 1)


class TestDecodeZiplist(unittest.TestCase):

    def test_round_trip(self):
        items = [b"hello", b"world"]
        zl = _make_ziplist(items)
        self.assertEqual(_decode_ziplist(zl), items)


class TestDecodeIntset(unittest.TestCase):

    def test_values(self):
        intset = _make_intset([5, 10, 15])
        result = _decode_intset(intset)
        self.assertEqual(set(result), {b"5", b"10", b"15"})


class TestDecodeListpack(unittest.TestCase):

    def test_string_elements(self):
        lp = _make_listpack([b"foo", b"bar"])
        result = _decode_listpack(lp)
        self.assertEqual(result, [b"foo", b"bar"])

    def test_empty_listpack(self):
        lp = _make_listpack([])
        result = _decode_listpack(lp)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
