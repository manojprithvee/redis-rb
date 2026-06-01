"""
Redis RDB file parser supporting RDB versions up to v10 (Redis 7.x).

RDB v10 adds: listpack-encoded sets, hash-listpack, zset-listpack,
quicklist2 (with listpack nodes), and stream listpacks v3.
"""

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------
RDB_OPCODE_AUX          = 0xFA
RDB_OPCODE_RESIZEDB     = 0xFB
RDB_OPCODE_EXPIRETIME_MS = 0xFC
RDB_OPCODE_EXPIRETIME   = 0xFD
RDB_OPCODE_SELECTDB     = 0xFE
RDB_OPCODE_EOF          = 0xFF

# ---------------------------------------------------------------------------
# Value type constants
# ---------------------------------------------------------------------------
RDB_TYPE_STRING             = 0
RDB_TYPE_LIST               = 1
RDB_TYPE_SET                = 2
RDB_TYPE_ZSET               = 3
RDB_TYPE_HASH               = 4
RDB_TYPE_ZSET_2             = 5
RDB_TYPE_MODULE             = 6
RDB_TYPE_MODULE_2           = 7
RDB_TYPE_HASH_ZIPMAP        = 9   # deprecated, v2
RDB_TYPE_LIST_ZIPLIST       = 10
RDB_TYPE_SET_INTSET         = 11
RDB_TYPE_ZSET_ZIPLIST       = 12
RDB_TYPE_HASH_ZIPLIST       = 13
RDB_TYPE_LIST_QUICKLIST     = 14
RDB_TYPE_STREAM_LISTPACKS   = 15
RDB_TYPE_HASH_LISTPACK      = 16  # v10
RDB_TYPE_ZSET_LISTPACK      = 17  # v10
RDB_TYPE_LIST_QUICKLIST_2   = 18  # v10
RDB_TYPE_SET_LISTPACK       = 19  # v10
RDB_TYPE_STREAM_LISTPACKS_3 = 20  # v10

# Length-encoding special types
RDB_ENC_INT8    = 0
RDB_ENC_INT16   = 1
RDB_ENC_INT32   = 2
RDB_ENC_LZF     = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RDBEntry:
    db: int
    key: bytes
    value: Any
    value_type: int
    expire_ms: Optional[int] = None  # epoch ms, None = no expiry


@dataclass
class RDBFile:
    version: int
    aux: Dict[bytes, bytes] = field(default_factory=dict)
    entries: List[RDBEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LZF decompression (pure-Python fallback)
# ---------------------------------------------------------------------------
def _lzf_decompress(data: bytes, expected_len: int) -> bytes:
    """Pure-Python LZF decompressor used when the lzf C extension is absent."""
    out = bytearray(expected_len)
    ip = 0
    op = 0
    while ip < len(data):
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:
            # literal run: ctrl+1 bytes
            length = ctrl + 1
            out[op:op + length] = data[ip:ip + length]
            ip += length
            op += length
        else:
            length = ctrl >> 5
            if length == 7:
                length += data[ip]
                ip += 1
            length += 2
            ref = op - ((ctrl & 0x1F) << 8) - data[ip] - 1
            ip += 1
            for i in range(length):
                out[op] = out[ref + i]
                op += 1
    return bytes(out)


def _decompress_lzf(compressed: bytes, uncompressed_len: int) -> bytes:
    try:
        import lzf as _lzf
        return _lzf.decompress(compressed, uncompressed_len)
    except ImportError:
        return _lzf_decompress(compressed, uncompressed_len)


# ---------------------------------------------------------------------------
# Low-level reader
# ---------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes):
        self._buf = data
        self._pos = 0

    def pos(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def read(self, n: int) -> bytes:
        chunk = self._buf[self._pos:self._pos + n]
        if len(chunk) < n:
            raise EOFError(f"Expected {n} bytes at pos {self._pos}, got {len(chunk)}")
        self._pos += n
        return chunk

    def read_byte(self) -> int:
        b = self._buf[self._pos]
        self._pos += 1
        return b

    def peek_byte(self) -> int:
        return self._buf[self._pos]

    # -- length-prefixed encoding -------------------------------------------
    def read_length(self) -> Tuple[int, bool]:
        """Returns (length_or_int_value, is_special).
        is_special=True means the value is an encoded integer type ID,
        not a byte length.
        """
        first = self.read_byte()
        enc_type = (first & 0xC0) >> 6
        if enc_type == 0:           # 6-bit length
            return first & 0x3F, False
        elif enc_type == 1:         # 14-bit length
            second = self.read_byte()
            return ((first & 0x3F) << 8) | second, False
        elif enc_type == 2:         # 32-bit or 64-bit length
            # In RDB >=7: big-endian 32-bit
            raw = self.read(4)
            return struct.unpack(">I", raw)[0], False
        else:                       # enc_type == 3 → special encoding
            return first & 0x3F, True

    def read_string(self) -> bytes:
        length, is_special = self.read_length()
        if not is_special:
            return self.read(length)
        # Special encodings
        if length == RDB_ENC_INT8:
            return str(struct.unpack("b", self.read(1))[0]).encode()
        elif length == RDB_ENC_INT16:
            return str(struct.unpack("<h", self.read(2))[0]).encode()
        elif length == RDB_ENC_INT32:
            return str(struct.unpack("<i", self.read(4))[0]).encode()
        elif length == RDB_ENC_LZF:
            clen, _ = self.read_length()
            ulen, _ = self.read_length()
            compressed = self.read(clen)
            return _decompress_lzf(compressed, ulen)
        else:
            raise ValueError(f"Unknown special string encoding: {length}")

    def read_double(self) -> float:
        length = self.read_byte()
        if length == 253:
            return float("nan")
        elif length == 254:
            return float("inf")
        elif length == 255:
            return float("-inf")
        return float(self.read(length))

    def read_double_binary(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def read_uint16_le(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def read_uint32_le(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_uint64_le(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]


# ---------------------------------------------------------------------------
# Listpack decoder (used by v10 types)
# ---------------------------------------------------------------------------
def _decode_listpack(data: bytes) -> List[bytes]:
    """Parse a Redis listpack blob and return its elements as bytes."""
    r = _Reader(data)
    _total_bytes = r.read_uint32_le()
    num_elements = r.read_uint16_le()
    result = []
    while r.peek_byte() != 0xFF:
        element = _lp_read_element(r)
        result.append(element)
    return result


def _lp_read_element(r: _Reader) -> bytes:
    first = r.read_byte()
    if first & 0x80 == 0:
        # 7-bit uint: 0xxxxxxx (0x00–0x7F)
        val = first & 0x7F
        _lp_skip_backlen(r)
        return str(val).encode()
    elif first & 0xC0 == 0x80:
        # 6-bit string: 10xxxxxx (0x80–0xBF), low 6 bits = length
        slen = first & 0x3F
        s = r.read(slen)
        _lp_skip_backlen(r)
        return s
    elif first & 0xE0 == 0xC0:
        # 13-bit signed int: 110xxxxx yyyyyyyy (0xC0–0xDF)
        second = r.read_byte()
        val = ((first & 0x1F) << 8) | second
        if val >= 0x1000:
            val -= 0x2000
        _lp_skip_backlen(r)
        return str(val).encode()
    elif first & 0xF0 == 0xE0:
        # 12-bit string: 1110xxxx xxxxxxxx (0xE0–0xEF)
        slen = ((first & 0x0F) << 8) | r.read_byte()
        s = r.read(slen)
        _lp_skip_backlen(r)
        return s
    elif first == 0xF1:
        val = struct.unpack("<h", r.read(2))[0]
        _lp_skip_backlen(r)
        return str(val).encode()
    elif first == 0xF2:
        b = r.read(3)
        val = struct.unpack("<i", b + (b'\xff' if b[2] & 0x80 else b'\x00'))[0]
        _lp_skip_backlen(r)
        return str(val).encode()
    elif first == 0xF3:
        val = struct.unpack("<i", r.read(4))[0]
        _lp_skip_backlen(r)
        return str(val).encode()
    elif first == 0xF4:
        val = struct.unpack("<q", r.read(8))[0]
        _lp_skip_backlen(r)
        return str(val).encode()
    else:
        raise ValueError(f"Unknown listpack encoding byte: 0x{first:02X} at pos {r.pos()-1}")


def _lp_skip_backlen(r: _Reader) -> None:
    """Skip the variable-length backlen field at the end of each listpack entry."""
    b = r.read_byte()
    while b & 0x80:
        b = r.read_byte()


# ---------------------------------------------------------------------------
# Ziplist decoder (legacy types)
# ---------------------------------------------------------------------------
def _decode_ziplist(data: bytes) -> List[bytes]:
    r = _Reader(data)
    _zlbytes = r.read_uint32_le()
    _zltail  = r.read_uint32_le()
    zllen    = r.read_uint16_le()
    result   = []
    for _ in range(zllen):
        entry = _zl_read_entry(r)
        if entry is None:
            break
        result.append(entry)
    return result


def _zl_read_entry(r: _Reader) -> Optional[bytes]:
    prevlen_byte = r.read_byte()
    if prevlen_byte == 0xFF:
        return None
    if prevlen_byte == 0xFE:
        r.read(4)  # 5-byte prevlen

    encoding = r.read_byte()
    if encoding >> 6 == 0:         # 6-bit string
        slen = encoding & 0x3F
        return r.read(slen)
    elif encoding >> 6 == 1:       # 14-bit string
        slen = ((encoding & 0x3F) << 8) | r.read_byte()
        return r.read(slen)
    elif encoding >> 6 == 2:       # 32-bit string
        slen = struct.unpack(">I", r.read(4))[0]
        return r.read(slen)
    elif encoding == 0xC0:
        return str(struct.unpack("<h", r.read(2))[0]).encode()
    elif encoding == 0xD0:
        return str(struct.unpack("<i", r.read(4))[0]).encode()
    elif encoding == 0xE0:
        return str(struct.unpack("<q", r.read(8))[0]).encode()
    elif encoding == 0xF0:
        b = r.read(3)
        val = struct.unpack("<i", b + b'\x00')[0]
        return str(val).encode()
    elif encoding == 0xFE:
        return str(struct.unpack("b", r.read(1))[0]).encode()
    elif encoding >= 0xF1 and encoding <= 0xFD:
        return str(encoding - 0xF1).encode()
    return None


# ---------------------------------------------------------------------------
# Intset decoder
# ---------------------------------------------------------------------------
def _decode_intset(data: bytes) -> List[bytes]:
    r = _Reader(data)
    encoding = r.read_uint32_le()
    length   = r.read_uint32_le()
    fmt = {2: "<h", 4: "<i", 8: "<q"}[encoding]
    size = encoding
    result = []
    for _ in range(length):
        val = struct.unpack(fmt, r.read(size))[0]
        result.append(str(val).encode())
    return result


# ---------------------------------------------------------------------------
# Type-specific value parsers
# ---------------------------------------------------------------------------
def _parse_string(r: _Reader) -> bytes:
    return r.read_string()


def _parse_list(r: _Reader) -> List[bytes]:
    length, _ = r.read_length()
    return [r.read_string() for _ in range(length)]


def _parse_set(r: _Reader) -> List[bytes]:
    length, _ = r.read_length()
    return [r.read_string() for _ in range(length)]


def _parse_zset(r: _Reader) -> List[Tuple[bytes, float]]:
    length, _ = r.read_length()
    result = []
    for _ in range(length):
        member = r.read_string()
        score  = r.read_double()
        result.append((member, score))
    return result


def _parse_zset2(r: _Reader) -> List[Tuple[bytes, float]]:
    length, _ = r.read_length()
    result = []
    for _ in range(length):
        member = r.read_string()
        score  = r.read_double_binary()
        result.append((member, score))
    return result


def _parse_hash(r: _Reader) -> Dict[bytes, bytes]:
    length, _ = r.read_length()
    return {r.read_string(): r.read_string() for _ in range(length)}


def _parse_list_ziplist(r: _Reader) -> List[bytes]:
    return _decode_ziplist(r.read_string())


def _parse_set_intset(r: _Reader) -> List[bytes]:
    return _decode_intset(r.read_string())


def _parse_zset_ziplist(r: _Reader) -> List[Tuple[bytes, float]]:
    elements = _decode_ziplist(r.read_string())
    return [(elements[i], float(elements[i + 1])) for i in range(0, len(elements), 2)]


def _parse_hash_ziplist(r: _Reader) -> Dict[bytes, bytes]:
    elements = _decode_ziplist(r.read_string())
    return {elements[i]: elements[i + 1] for i in range(0, len(elements), 2)}


def _parse_list_quicklist(r: _Reader) -> List[bytes]:
    num_nodes, _ = r.read_length()
    result = []
    for _ in range(num_nodes):
        result.extend(_decode_ziplist(r.read_string()))
    return result


def _parse_list_quicklist2(r: _Reader) -> List[bytes]:
    """Quicklist v2: nodes can be ziplist or listpack."""
    num_nodes, _ = r.read_length()
    result = []
    for _ in range(num_nodes):
        data = r.read_string()
        container, _ = r.read_length()  # 1=plain, 2=ziplist/listpack
        if container == 2:
            # Redis 7.x uses listpack nodes
            try:
                result.extend(_decode_listpack(data))
            except Exception:
                result.extend(_decode_ziplist(data))
        else:
            result.append(data)
    return result


def _parse_hash_listpack(r: _Reader) -> Dict[bytes, bytes]:
    elements = _decode_listpack(r.read_string())
    return {elements[i]: elements[i + 1] for i in range(0, len(elements), 2)}


def _parse_zset_listpack(r: _Reader) -> List[Tuple[bytes, float]]:
    elements = _decode_listpack(r.read_string())
    return [(elements[i], float(elements[i + 1])) for i in range(0, len(elements), 2)]


def _parse_set_listpack(r: _Reader) -> List[bytes]:
    return _decode_listpack(r.read_string())


def _parse_stream_listpacks(r: _Reader) -> Dict:
    """Parse stream type (v15). Returns a dict with raw structure."""
    num_listpacks, _ = r.read_length()
    listpacks = []
    for _ in range(num_listpacks):
        master_id = r.read_string()  # master entry ID
        lp_data   = r.read_string()  # listpack blob
        listpacks.append({"master_id": master_id, "data": lp_data})

    length, _       = r.read_length()
    last_ms         = r.read_uint64_le()
    last_seq        = r.read_uint64_le()
    first_ms        = r.read_uint64_le()
    first_seq       = r.read_uint64_le()
    max_del_ms      = r.read_uint64_le()
    max_del_seq     = r.read_uint64_le()
    entries_added, _ = r.read_length()

    num_cgroups, _ = r.read_length()
    cgroups = []
    for _ in range(num_cgroups):
        cg_name   = r.read_string()
        last_del_ms  = r.read_uint64_le()
        last_del_seq = r.read_uint64_le()
        entries_read, _ = r.read_length()

        num_pel, _ = r.read_length()
        pel = []
        for _ in range(num_pel):
            eid = r.read(16)
            ts  = r.read_uint64_le()
            dc  = r.read_uint64_le()
            count, _ = r.read_length()
            pel.append({"id": eid, "ts": ts, "delivery_count": count})

        num_consumers, _ = r.read_length()
        consumers = []
        for _ in range(num_consumers):
            cname = r.read_string()
            seen  = r.read_uint64_le()
            active = r.read_uint64_le()
            num_cpel, _ = r.read_length()
            cpel_ids = [r.read(16) for _ in range(num_cpel)]
            consumers.append({"name": cname, "seen_time": seen, "pel_ids": cpel_ids})

        cgroups.append({
            "name": cg_name,
            "pel": pel,
            "consumers": consumers,
        })

    return {
        "listpacks": listpacks,
        "length": length,
        "last_id": (last_ms, last_seq),
        "first_id": (first_ms, first_seq),
        "cgroups": cgroups,
    }


def _parse_stream_listpacks3(r: _Reader) -> Dict:
    """Stream v3 (RDB type 20, Redis 7.4+). Extended stream metadata."""
    return _parse_stream_listpacks(r)


def _parse_module(r: _Reader) -> bytes:
    """Skip/collect unknown module data."""
    r.read_string()  # module name
    length, _ = r.read_length()
    return r.read(length)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
_TYPE_PARSERS = {
    RDB_TYPE_STRING:             _parse_string,
    RDB_TYPE_LIST:               _parse_list,
    RDB_TYPE_SET:                _parse_set,
    RDB_TYPE_ZSET:               _parse_zset,
    RDB_TYPE_ZSET_2:             _parse_zset2,
    RDB_TYPE_HASH:               _parse_hash,
    RDB_TYPE_LIST_ZIPLIST:       _parse_list_ziplist,
    RDB_TYPE_SET_INTSET:         _parse_set_intset,
    RDB_TYPE_ZSET_ZIPLIST:       _parse_zset_ziplist,
    RDB_TYPE_HASH_ZIPLIST:       _parse_hash_ziplist,
    RDB_TYPE_LIST_QUICKLIST:     _parse_list_quicklist,
    RDB_TYPE_LIST_QUICKLIST_2:   _parse_list_quicklist2,
    RDB_TYPE_HASH_LISTPACK:      _parse_hash_listpack,
    RDB_TYPE_ZSET_LISTPACK:      _parse_zset_listpack,
    RDB_TYPE_SET_LISTPACK:       _parse_set_listpack,
    RDB_TYPE_STREAM_LISTPACKS:   _parse_stream_listpacks,
    RDB_TYPE_STREAM_LISTPACKS_3: _parse_stream_listpacks3,
}


def parse_rdb(data: bytes) -> RDBFile:
    """Parse a full RDB file from bytes and return an RDBFile object."""
    r = _Reader(data)

    magic = r.read(5)
    if magic != b"REDIS":
        raise ValueError(f"Not an RDB file (bad magic: {magic!r})")

    version = int(r.read(4))
    if version > 10:
        raise ValueError(f"Unsupported RDB version: {version} (max supported: 10)")

    rdb = RDBFile(version=version)
    current_db    = 0
    expire_ms: Optional[int] = None

    while True:
        opcode = r.read_byte()

        if opcode == RDB_OPCODE_EOF:
            # Optional 8-byte CRC64 checksum
            if r.remaining() >= 8:
                _checksum = r.read(8)
            break

        elif opcode == RDB_OPCODE_AUX:
            key   = r.read_string()
            value = r.read_string()
            rdb.aux[key] = value

        elif opcode == RDB_OPCODE_RESIZEDB:
            _db_size, _   = r.read_length()
            _expire_size, _ = r.read_length()

        elif opcode == RDB_OPCODE_SELECTDB:
            current_db, _ = r.read_length()

        elif opcode == RDB_OPCODE_EXPIRETIME_MS:
            expire_ms = struct.unpack("<Q", r.read(8))[0]

        elif opcode == RDB_OPCODE_EXPIRETIME:
            expire_ms = struct.unpack("<I", r.read(4))[0] * 1000

        else:
            # opcode is actually the value type byte
            value_type = opcode
            key = r.read_string()

            parser = _TYPE_PARSERS.get(value_type)
            if parser is None:
                raise ValueError(
                    f"Unknown RDB value type: {value_type} for key {key!r}"
                )

            value = parser(r)
            rdb.entries.append(RDBEntry(
                db=current_db,
                key=key,
                value=value,
                value_type=value_type,
                expire_ms=expire_ms,
            ))
            expire_ms = None  # reset after consuming

    return rdb


def parse_rdb_file(path: str) -> RDBFile:
    """Parse an RDB file from disk."""
    with open(path, "rb") as f:
        return parse_rdb(f.read())


def iter_entries(path: str) -> Iterator[RDBEntry]:
    """Lazily iterate over RDB entries without loading all into memory.
    NOTE: Current implementation loads the file once; true streaming would
    require a stateful reader passed entry-by-entry.
    """
    rdb = parse_rdb_file(path)
    yield from rdb.entries
