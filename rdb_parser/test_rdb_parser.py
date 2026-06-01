"""Tests for the RDB v10 parser (Redis 7.0 / rdb.h RDB_VERSION 10)."""

import struct
import unittest
from rdb_parser import (
    parse_rdb, RDBFile, RDBEntry,
    RDB_TYPE_STRING, RDB_TYPE_LIST, RDB_TYPE_SET, RDB_TYPE_ZSET,
    RDB_TYPE_ZSET_2, RDB_TYPE_HASH, RDB_TYPE_LIST_ZIPLIST,
    RDB_TYPE_SET_INTSET, RDB_TYPE_ZSET_ZIPLIST, RDB_TYPE_HASH_ZIPLIST,
    RDB_TYPE_LIST_QUICKLIST, RDB_TYPE_LIST_QUICKLIST_2,
    RDB_TYPE_HASH_LISTPACK, RDB_TYPE_ZSET_LISTPACK,
    RDB_TYPE_STREAM_LISTPACKS, RDB_TYPE_STREAM_LISTPACKS_2,
    _decode_ziplist, _decode_intset, _decode_listpack,
    RDB_OPCODE_AUX, RDB_OPCODE_SELECTDB, RDB_OPCODE_EXPIRETIME_MS,
    RDB_OPCODE_EXPIRETIME, RDB_OPCODE_RESIZEDB, RDB_OPCODE_EOF,
    RDB_OPCODE_IDLE, RDB_OPCODE_FREQ,
    RDB_OPCODE_FUNCTION2, RDB_OPCODE_FUNCTION,
    RDB_32BITLEN, RDB_64BITLEN,
)


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def _len_encode(n: int) -> bytes:
    if n <= 63:
        return bytes([n])
    elif n <= 16383:
        return bytes([0x40 | (n >> 8), n & 0xFF])
    else:
        return bytes([RDB_32BITLEN]) + struct.pack(">I", n)


def _str_encode(s: bytes) -> bytes:
    return _len_encode(len(s)) + s


def _make_rdb(version: int, body: bytes) -> bytes:
    return b"REDIS" + f"{version:04d}".encode() + body + bytes([RDB_OPCODE_EOF]) + b'\x00' * 8


def _entry(vtype: int, key: bytes, value_blob: bytes,
           expire_ms: int = None, lru_idle: int = None, lfu_freq: int = None) -> bytes:
    parts = b""
    if expire_ms is not None:
        parts += bytes([RDB_OPCODE_EXPIRETIME_MS]) + struct.pack("<Q", expire_ms)
    if lru_idle is not None:
        parts += bytes([RDB_OPCODE_IDLE]) + _len_encode(lru_idle)
    if lfu_freq is not None:
        parts += bytes([RDB_OPCODE_FREQ]) + bytes([lfu_freq])
    parts += bytes([vtype]) + _str_encode(key) + value_blob
    return parts


# ---------------------------------------------------------------------------
# Ziplist builder
# ---------------------------------------------------------------------------

def _zl_str_entry(s: bytes) -> bytes:
    prevlen = b'\x00'
    enc = bytes([len(s)]) if len(s) <= 63 else bytes([0x40, len(s)])
    return prevlen + enc + s


def _make_ziplist(items: list) -> bytes:
    entries = b"".join(_zl_str_entry(i if isinstance(i, bytes) else str(i).encode())
                       for i in items) + b'\xFF'
    total = 4 + 4 + 2 + len(entries)
    return struct.pack("<I", total) + struct.pack("<I", 4 + 4 + 2) + struct.pack("<H", len(items)) + entries


# ---------------------------------------------------------------------------
# Intset builder
# ---------------------------------------------------------------------------

def _make_intset(values: list, encoding: int = 8) -> bytes:
    fmt = {2: "<h", 4: "<i", 8: "<q"}[encoding]
    body = b"".join(struct.pack(fmt, v) for v in sorted(values))
    return struct.pack("<I", encoding) + struct.pack("<I", len(values)) + body


# ---------------------------------------------------------------------------
# Listpack builder
# Elements are encoded as:
#   6-bit string:  10xxxxxx + data + backlen(1+len)
#   7-bit uint:    0xxxxxxx + backlen(1)
# ---------------------------------------------------------------------------

def _lp_str6(s: bytes) -> bytes:
    assert len(s) <= 63
    return bytes([0x80 | len(s)]) + s + bytes([1 + len(s)])


def _lp_uint7(v: int) -> bytes:
    assert 0 <= v <= 127
    return bytes([v & 0x7F]) + b'\x01'


def _make_listpack(items: list) -> bytes:
    entries = b""
    for item in items:
        if isinstance(item, int) and 0 <= item <= 127:
            entries += _lp_uint7(item)
        else:
            b = item if isinstance(item, bytes) else str(item).encode()
            entries += _lp_str6(b)
    entries += b'\xFF'
    total = 4 + 2 + len(entries)
    return struct.pack("<I", total) + struct.pack("<H", len(items)) + entries


# ---------------------------------------------------------------------------
# Stream builder helpers
# ---------------------------------------------------------------------------

def _make_stream_v1(listpacks, length, last_ms, last_seq, cgroups=None) -> bytes:
    """Build a minimal STREAM_LISTPACKS (type 15) blob."""
    cgroups = cgroups or []
    blob = _len_encode(len(listpacks))
    for master_id, lp_data in listpacks:
        blob += _str_encode(master_id) + _str_encode(lp_data)
    blob += _len_encode(length)
    blob += _len_encode(last_ms) + _len_encode(last_seq)
    blob += _len_encode(len(cgroups))
    for cg in cgroups:
        blob += _str_encode(cg["name"])
        blob += _len_encode(cg["last_ms"]) + _len_encode(cg["last_seq"])
        blob += _len_encode(len(cg.get("pel", [])))
        for pe in cg.get("pel", []):
            blob += pe["id"] + struct.pack("<Q", pe["delivery_time"]) + _len_encode(pe["delivery_count"])
        blob += _len_encode(len(cg.get("consumers", [])))
        for c in cg.get("consumers", []):
            blob += _str_encode(c["name"]) + struct.pack("<Q", c["seen_time"])
            blob += _len_encode(0)  # empty consumer PEL
    return blob


def _make_stream_v2(listpacks, length, last_ms, last_seq,
                    first_ms, first_seq, max_del_ms, max_del_seq,
                    entries_added, cgroups=None) -> bytes:
    """Build a minimal STREAM_LISTPACKS_2 (type 19) blob."""
    cgroups = cgroups or []
    blob = _len_encode(len(listpacks))
    for master_id, lp_data in listpacks:
        blob += _str_encode(master_id) + _str_encode(lp_data)
    blob += _len_encode(length)
    blob += _len_encode(last_ms) + _len_encode(last_seq)
    blob += _len_encode(first_ms) + _len_encode(first_seq)
    blob += _len_encode(max_del_ms) + _len_encode(max_del_seq)
    blob += _len_encode(entries_added)
    blob += _len_encode(len(cgroups))
    for cg in cgroups:
        blob += _str_encode(cg["name"])
        blob += _len_encode(cg["last_ms"]) + _len_encode(cg["last_seq"])
        blob += _len_encode(cg.get("entries_read", 0))  # v2-only
        blob += _len_encode(len(cg.get("pel", [])))
        for pe in cg.get("pel", []):
            blob += pe["id"] + struct.pack("<Q", pe["delivery_time"]) + _len_encode(pe["delivery_count"])
        blob += _len_encode(len(cg.get("consumers", [])))
        for c in cg.get("consumers", []):
            blob += _str_encode(c["name"]) + struct.pack("<Q", c["seen_time"])
            blob += _len_encode(0)
    return blob


# ===========================================================================
# Tests
# ===========================================================================

class TestMagicAndVersion(unittest.TestCase):

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            parse_rdb(b"WRONG0010" + bytes([RDB_OPCODE_EOF]) + b'\x00' * 8)

    def test_version_too_high(self):
        with self.assertRaises(ValueError):
            parse_rdb(_make_rdb(11, b""))

    def test_version_10(self):
        self.assertEqual(parse_rdb(_make_rdb(10, b"")).version, 10)

    def test_version_9(self):
        self.assertEqual(parse_rdb(_make_rdb(9, b"")).version, 9)


class TestAuxFields(unittest.TestCase):

    def test_single(self):
        body = bytes([RDB_OPCODE_AUX]) + _str_encode(b"redis-ver") + _str_encode(b"7.0.0")
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.aux[b"redis-ver"], b"7.0.0")

    def test_multiple(self):
        body = (bytes([RDB_OPCODE_AUX]) + _str_encode(b"a") + _str_encode(b"1") +
                bytes([RDB_OPCODE_AUX]) + _str_encode(b"b") + _str_encode(b"2"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(len(rdb.aux), 2)


class TestFunctionOpcodes(unittest.TestCase):

    def test_function2_stored(self):
        payload = b"<serialised-lua-lib>"
        body = bytes([RDB_OPCODE_FUNCTION2]) + _str_encode(payload)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertIn(payload, rdb.functions)

    def test_function_stored(self):
        payload = b"<old-lib>"
        body = bytes([RDB_OPCODE_FUNCTION]) + _str_encode(payload)
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertIn(payload, rdb.functions)


class TestSelectDB(unittest.TestCase):

    def test_db_number(self):
        body = bytes([RDB_OPCODE_SELECTDB, 3]) + _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v"))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(rdb.entries[0].db, 3)


class TestResizeDB(unittest.TestCase):

    def test_resizedb_consumed(self):
        body = (bytes([RDB_OPCODE_SELECTDB, 0]) +
                bytes([RDB_OPCODE_RESIZEDB]) + _len_encode(5) + _len_encode(2) +
                _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v")))
        rdb = parse_rdb(_make_rdb(10, body))
        self.assertEqual(len(rdb.entries), 1)


class TestExpiry(unittest.TestCase):

    def test_expiretime_ms(self):
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v"), expire_ms=1_700_000_000_000)
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].expire_ms, 1_700_000_000_000)

    def test_expiretime_seconds(self):
        ts = 1_700_000_000
        body = (bytes([RDB_OPCODE_EXPIRETIME]) + struct.pack("<I", ts) +
                bytes([RDB_TYPE_STRING]) + _str_encode(b"k") + _str_encode(b"v"))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].expire_ms, ts * 1000)

    def test_no_expiry(self):
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v"))
        self.assertIsNone(parse_rdb(_make_rdb(10, body)).entries[0].expire_ms)


class TestIdleAndFreq(unittest.TestCase):

    def test_idle_attached_to_entry(self):
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v"), lru_idle=3600)
        e = parse_rdb(_make_rdb(10, body)).entries[0]
        self.assertEqual(e.lru_idle, 3600)

    def test_freq_attached_to_entry(self):
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b"v"), lfu_freq=42)
        e = parse_rdb(_make_rdb(10, body)).entries[0]
        self.assertEqual(e.lfu_freq, 42)

    def test_idle_resets_after_entry(self):
        body = (bytes([RDB_OPCODE_IDLE]) + _len_encode(100) +
                bytes([RDB_TYPE_STRING]) + _str_encode(b"k1") + _str_encode(b"v1") +
                bytes([RDB_TYPE_STRING]) + _str_encode(b"k2") + _str_encode(b"v2"))
        entries = parse_rdb(_make_rdb(10, body)).entries
        self.assertEqual(entries[0].lru_idle, 100)
        self.assertIsNone(entries[1].lru_idle)


class TestLengthEncoding(unittest.TestCase):

    def test_6bit(self):
        # a 5-byte string encoded with 6-bit length
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b"hello"))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, b"hello")

    def test_64bit_length(self):
        # Construct a string length encoded with RDB_64BITLEN (0x81 + 8 BE bytes)
        s = b"x" * 200
        length_field = bytes([RDB_64BITLEN]) + struct.pack(">Q", len(s))
        value_blob = length_field + s
        body = bytes([RDB_TYPE_STRING]) + _str_encode(b"bigkey") + value_blob
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, s)

    def test_32bit_length(self):
        s = b"y" * 300
        length_field = bytes([RDB_32BITLEN]) + struct.pack(">I", len(s))
        value_blob = length_field + s
        body = bytes([RDB_TYPE_STRING]) + _str_encode(b"k") + value_blob
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, s)


class TestStringType(unittest.TestCase):

    def test_simple(self):
        body = _entry(RDB_TYPE_STRING, b"hello", _str_encode(b"world"))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, b"world")

    def test_empty(self):
        body = _entry(RDB_TYPE_STRING, b"k", _str_encode(b""))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, b"")


class TestListType(unittest.TestCase):

    def test_plain_list(self):
        items = [b"a", b"b", b"c"]
        blob = _len_encode(3) + b"".join(_str_encode(i) for i in items)
        body = _entry(RDB_TYPE_LIST, b"l", blob)
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, items)


class TestSetType(unittest.TestCase):

    def test_plain_set(self):
        items = [b"x", b"y"]
        blob = _len_encode(2) + b"".join(_str_encode(i) for i in items)
        body = _entry(RDB_TYPE_SET, b"s", blob)
        self.assertEqual(set(parse_rdb(_make_rdb(10, body)).entries[0].value), {b"x", b"y"})


class TestHashType(unittest.TestCase):

    def test_plain_hash(self):
        blob = _len_encode(2) + _str_encode(b"f1") + _str_encode(b"v1") + _str_encode(b"f2") + _str_encode(b"v2")
        body = _entry(RDB_TYPE_HASH, b"h", blob)
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, {b"f1": b"v1", b"f2": b"v2"})


class TestZSetTypes(unittest.TestCase):

    def _score_str(self, score):
        s = str(score).encode()
        return bytes([len(s)]) + s

    def test_zset_v1(self):
        blob = _len_encode(1) + _str_encode(b"m") + self._score_str(1.5)
        body = _entry(RDB_TYPE_ZSET, b"z", blob)
        pairs = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertAlmostEqual(pairs[0][1], 1.5)

    def test_zset_v2_binary(self):
        blob = _len_encode(1) + _str_encode(b"m") + struct.pack("<d", 3.14)
        body = _entry(RDB_TYPE_ZSET_2, b"z2", blob)
        pairs = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertAlmostEqual(pairs[0][1], 3.14)


class TestZiplistTypes(unittest.TestCase):

    def test_list_ziplist(self):
        zl = _make_ziplist([b"a", b"b"])
        body = _entry(RDB_TYPE_LIST_ZIPLIST, b"l", _str_encode(zl))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, [b"a", b"b"])

    def test_hash_ziplist(self):
        zl = _make_ziplist([b"k", b"v"])
        body = _entry(RDB_TYPE_HASH_ZIPLIST, b"h", _str_encode(zl))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, {b"k": b"v"})

    def test_zset_ziplist(self):
        zl = _make_ziplist([b"m", b"2.5"])
        body = _entry(RDB_TYPE_ZSET_ZIPLIST, b"z", _str_encode(zl))
        pairs = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertAlmostEqual(pairs[0][1], 2.5)


class TestIntset(unittest.TestCase):

    def test_intset_64bit(self):
        is_ = _make_intset([1, 2, 3], encoding=8)
        body = _entry(RDB_TYPE_SET_INTSET, b"s", _str_encode(is_))
        self.assertEqual(set(parse_rdb(_make_rdb(10, body)).entries[0].value), {b"1", b"2", b"3"})

    def test_intset_16bit(self):
        is_ = _make_intset([10, 20], encoding=2)
        body = _entry(RDB_TYPE_SET_INTSET, b"s", _str_encode(is_))
        self.assertEqual(set(parse_rdb(_make_rdb(10, body)).entries[0].value), {b"10", b"20"})


class TestListpackTypes(unittest.TestCase):

    def test_hash_listpack(self):
        lp = _make_listpack([b"k1", b"v1", b"k2", b"v2"])
        body = _entry(RDB_TYPE_HASH_LISTPACK, b"h", _str_encode(lp))
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, {b"k1": b"v1", b"k2": b"v2"})

    def test_zset_listpack(self):
        lp = _make_listpack([b"m", b"3"])
        body = _entry(RDB_TYPE_ZSET_LISTPACK, b"z", _str_encode(lp))
        pairs = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertAlmostEqual(pairs[0][1], 3.0)


class TestQuicklist(unittest.TestCase):

    def test_quicklist_v1(self):
        zl = _make_ziplist([b"x", b"y"])
        blob = _len_encode(1) + _str_encode(zl)
        body = _entry(RDB_TYPE_LIST_QUICKLIST, b"l", blob)
        self.assertEqual(parse_rdb(_make_rdb(10, body)).entries[0].value, [b"x", b"y"])

    def test_quicklist_v2_listpack_node(self):
        lp = _make_listpack([b"a", b"b"])
        blob = _len_encode(1) + _str_encode(lp) + _len_encode(2)  # container=2 (listpack)
        body = _entry(RDB_TYPE_LIST_QUICKLIST_2, b"l", blob)
        result = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertIn(b"a", result)
        self.assertIn(b"b", result)

    def test_quicklist_v2_plain_node(self):
        blob = _len_encode(1) + _str_encode(b"rawdata") + _len_encode(1)  # container=1 (plain)
        body = _entry(RDB_TYPE_LIST_QUICKLIST_2, b"l", blob)
        result = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(result, [b"rawdata"])


class TestStreamListpacksV1(unittest.TestCase):

    def test_empty_stream(self):
        blob = _make_stream_v1([], length=0, last_ms=0, last_seq=0)
        body = _entry(RDB_TYPE_STREAM_LISTPACKS, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(val["length"], 0)
        self.assertEqual(val["last_id"], (0, 0))
        self.assertEqual(val["cgroups"], [])

    def test_stream_with_metadata(self):
        blob = _make_stream_v1([], length=5, last_ms=1000, last_seq=2)
        body = _entry(RDB_TYPE_STREAM_LISTPACKS, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(val["length"], 5)
        self.assertEqual(val["last_id"], (1000, 2))

    def test_stream_with_cgroup(self):
        cgroups = [{"name": b"grp", "last_ms": 999, "last_seq": 1}]
        blob = _make_stream_v1([], length=0, last_ms=0, last_seq=0, cgroups=cgroups)
        body = _entry(RDB_TYPE_STREAM_LISTPACKS, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(len(val["cgroups"]), 1)
        self.assertEqual(val["cgroups"][0]["name"], b"grp")
        # v1 cgroup should have no entries_read
        self.assertIsNone(val["cgroups"][0]["entries_read"])

    def test_stream_v1_has_no_first_id_key(self):
        blob = _make_stream_v1([], length=0, last_ms=0, last_seq=0)
        body = _entry(RDB_TYPE_STREAM_LISTPACKS, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertNotIn("first_id", val)
        self.assertNotIn("entries_added", val)


class TestStreamListpacksV2(unittest.TestCase):

    def test_empty_stream_v2(self):
        blob = _make_stream_v2(
            [], length=0, last_ms=0, last_seq=0,
            first_ms=0, first_seq=0, max_del_ms=0, max_del_seq=0,
            entries_added=0,
        )
        body = _entry(RDB_TYPE_STREAM_LISTPACKS_2, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(val["length"], 0)
        self.assertEqual(val["first_id"], (0, 0))
        self.assertEqual(val["max_deleted_entry_id"], (0, 0))
        self.assertEqual(val["entries_added"], 0)

    def test_stream_v2_metadata(self):
        blob = _make_stream_v2(
            [], length=10, last_ms=2000, last_seq=5,
            first_ms=100, first_seq=0, max_del_ms=50, max_del_seq=0,
            entries_added=15,
        )
        body = _entry(RDB_TYPE_STREAM_LISTPACKS_2, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(val["length"], 10)
        self.assertEqual(val["last_id"], (2000, 5))
        self.assertEqual(val["first_id"], (100, 0))
        self.assertEqual(val["entries_added"], 15)

    def test_stream_v2_cgroup_entries_read(self):
        cgroups = [{"name": b"g", "last_ms": 1, "last_seq": 0, "entries_read": 7}]
        blob = _make_stream_v2(
            [], length=0, last_ms=0, last_seq=0,
            first_ms=0, first_seq=0, max_del_ms=0, max_del_seq=0,
            entries_added=0, cgroups=cgroups,
        )
        body = _entry(RDB_TYPE_STREAM_LISTPACKS_2, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        self.assertEqual(val["cgroups"][0]["entries_read"], 7)

    def test_stream_v2_pel_and_consumer(self):
        raw_id = b'\x00' * 16
        pel = [{"id": raw_id, "delivery_time": 9999, "delivery_count": 2}]
        consumers = [{"name": b"alice", "seen_time": 12345}]
        cgroups = [{"name": b"g", "last_ms": 0, "last_seq": 0,
                    "entries_read": 0, "pel": pel, "consumers": consumers}]
        blob = _make_stream_v2(
            [], length=0, last_ms=0, last_seq=0,
            first_ms=0, first_seq=0, max_del_ms=0, max_del_seq=0,
            entries_added=0, cgroups=cgroups,
        )
        body = _entry(RDB_TYPE_STREAM_LISTPACKS_2, b"s", blob)
        val = parse_rdb(_make_rdb(10, body)).entries[0].value
        cg = val["cgroups"][0]
        self.assertEqual(cg["pel"][0]["delivery_count"], 2)
        self.assertEqual(cg["consumers"][0]["name"], b"alice")
        self.assertEqual(cg["consumers"][0]["seen_time"], 12345)


class TestMultipleDatabases(unittest.TestCase):

    def test_multi_db(self):
        body = (bytes([RDB_OPCODE_SELECTDB, 0]) +
                _entry(RDB_TYPE_STRING, b"a", _str_encode(b"1")) +
                bytes([RDB_OPCODE_SELECTDB, 1]) +
                _entry(RDB_TYPE_STRING, b"b", _str_encode(b"2")))
        entries = parse_rdb(_make_rdb(10, body)).entries
        self.assertEqual(entries[0].db, 0)
        self.assertEqual(entries[1].db, 1)


class TestDecodeHelpers(unittest.TestCase):

    def test_decode_ziplist(self):
        self.assertEqual(_decode_ziplist(_make_ziplist([b"p", b"q"])), [b"p", b"q"])

    def test_decode_intset(self):
        self.assertEqual(set(_decode_intset(_make_intset([3, 1, 2]))), {b"1", b"2", b"3"})

    def test_decode_listpack_strings(self):
        self.assertEqual(_decode_listpack(_make_listpack([b"foo", b"bar"])), [b"foo", b"bar"])

    def test_decode_listpack_empty(self):
        self.assertEqual(_decode_listpack(_make_listpack([])), [])

    def test_decode_listpack_uint7(self):
        self.assertEqual(_decode_listpack(_make_listpack([5, 127])), [b"5", b"127"])


if __name__ == "__main__":
    unittest.main()
