"""
Redis RDB file parser supporting RDB versions up to v10 (Redis 7.0).

RDB v10 (Redis 7.0) adds on top of v9:
  HASH_LISTPACK (16), ZSET_LISTPACK (17), LIST_QUICKLIST_2 (18),
  STREAM_LISTPACKS_2 (19) with first_id / max_deleted_entry_id /
  entries_added / cgroup entries_read,
  and the FUNCTION2 / FUNCTION / MODULE_AUX opcodes (245-247).

Type reference (rdb.h, Redis 7.0):
  0  STRING          9  HASH_ZIPMAP (deprecated)
  1  LIST           10  LIST_ZIPLIST
  2  SET            11  SET_INTSET
  3  ZSET           12  ZSET_ZIPLIST
  4  HASH           13  HASH_ZIPLIST
  5  ZSET_2         14  LIST_QUICKLIST
  6  MODULE         15  STREAM_LISTPACKS
  7  MODULE_2       16  HASH_LISTPACK
                    17  ZSET_LISTPACK
                    18  LIST_QUICKLIST_2
                    19  STREAM_LISTPACKS_2
"""

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Opcodes (Redis 7.0 / RDB v10)
# ---------------------------------------------------------------------------
RDB_OPCODE_FUNCTION2     = 245  # serialised Lua function library (v10)
RDB_OPCODE_FUNCTION      = 246  # RC1/RC2 function library (v10)
RDB_OPCODE_MODULE_AUX    = 247  # module auxiliary data
RDB_OPCODE_IDLE          = 248  # LRU idle time for next key
RDB_OPCODE_FREQ          = 249  # LFU frequency byte for next key
RDB_OPCODE_AUX           = 250  # 0xFA – auxiliary field (redis-ver, etc.)
RDB_OPCODE_RESIZEDB      = 251  # 0xFB – hash table resize hint
RDB_OPCODE_EXPIRETIME_MS = 252  # 0xFC – expiry in ms
RDB_OPCODE_EXPIRETIME    = 253  # 0xFD – expiry in seconds (legacy)
RDB_OPCODE_SELECTDB      = 254  # 0xFE – database selector
RDB_OPCODE_EOF           = 255  # 0xFF – end of file + optional CRC64

# ---------------------------------------------------------------------------
# Value type constants (RDB v10)
# ---------------------------------------------------------------------------
RDB_TYPE_STRING             = 0
RDB_TYPE_LIST               = 1
RDB_TYPE_SET                = 2
RDB_TYPE_ZSET               = 3
RDB_TYPE_HASH               = 4
RDB_TYPE_ZSET_2             = 5   # binary-encoded double scores
RDB_TYPE_MODULE             = 6
RDB_TYPE_MODULE_2           = 7   # module with parsing annotations
RDB_TYPE_HASH_ZIPMAP        = 9   # deprecated since Redis 2.6
RDB_TYPE_LIST_ZIPLIST       = 10
RDB_TYPE_SET_INTSET         = 11
RDB_TYPE_ZSET_ZIPLIST       = 12
RDB_TYPE_HASH_ZIPLIST       = 13
RDB_TYPE_LIST_QUICKLIST     = 14
RDB_TYPE_STREAM_LISTPACKS   = 15  # Redis 5.0
RDB_TYPE_HASH_LISTPACK      = 16  # Redis 7.0 (RDB v10)
RDB_TYPE_ZSET_LISTPACK      = 17  # Redis 7.0
RDB_TYPE_LIST_QUICKLIST_2   = 18  # Redis 7.0
RDB_TYPE_STREAM_LISTPACKS_2 = 19  # Redis 7.0 – extended stream metadata

# Length-encoding special sub-types (enc_type == 3)
RDB_ENC_INT8  = 0
RDB_ENC_INT16 = 1
RDB_ENC_INT32 = 2
RDB_ENC_LZF   = 3

# Length-encoding enc_type == 2 first-byte values
RDB_32BITLEN = 0x80
RDB_64BITLEN = 0x81


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RDBEntry:
    db: int
    key: bytes
    value: Any
    value_type: int
    expire_ms: Optional[int] = None   # epoch ms, None = no expiry
    lru_idle: Optional[int] = None    # LRU idle seconds (IDLE opcode)
    lfu_freq: Optional[int] = None    # LFU frequency byte (FREQ opcode)


@dataclass
class RDBFile:
    version: int
    aux: Dict[bytes, bytes] = field(default_factory=dict)
    functions: List[bytes] = field(default_factory=list)  # FUNCTION2 payloads
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

    # -- length encoding (RDB_6BITLEN / RDB_14BITLEN / RDB_32BITLEN /
    #                     RDB_64BITLEN / RDB_ENCVAL)
    def read_length(self) -> Tuple[int, bool]:
        """Return (value, is_special).

        is_special=True  → value is a special-encoding type ID (not a length).
        is_special=False → value is a plain byte count.
        """
        first = self.read_byte()
        enc_type = (first & 0xC0) >> 6

        if enc_type == 0:       # RDB_6BITLEN: 6-bit length
            return first & 0x3F, False

        elif enc_type == 1:     # RDB_14BITLEN: 14-bit length
            second = self.read_byte()
            return ((first & 0x3F) << 8) | second, False

        elif enc_type == 2:     # RDB_32BITLEN or RDB_64BITLEN
            if first == RDB_32BITLEN:
                return struct.unpack(">I", self.read(4))[0], False
            elif first == RDB_64BITLEN:
                return struct.unpack(">Q", self.read(8))[0], False
            else:
                raise ValueError(f"Unknown length-encoding first byte: 0x{first:02X}")

        else:                   # enc_type == 3 → RDB_ENCVAL (special)
            return first & 0x3F, True

    def read_string(self) -> bytes:
        length, is_special = self.read_length()
        if not is_special:
            return self.read(length)
        if length == RDB_ENC_INT8:
            return str(struct.unpack("b", self.read(1))[0]).encode()
        elif length == RDB_ENC_INT16:
            return str(struct.unpack("<h", self.read(2))[0]).encode()
        elif length == RDB_ENC_INT32:
            return str(struct.unpack("<i", self.read(4))[0]).encode()
        elif length == RDB_ENC_LZF:
            clen, _ = self.read_length()
            ulen, _ = self.read_length()
            return _decompress_lzf(self.read(clen), ulen)
        else:
            raise ValueError(f"Unknown special string encoding: {length}")

    def read_double(self) -> float:
        """Legacy text-encoded double (RDB_TYPE_ZSET)."""
        length = self.read_byte()
        if length == 253:
            return float("nan")
        elif length == 254:
            return float("inf")
        elif length == 255:
            return float("-inf")
        return float(self.read(length))

    def read_double_binary(self) -> float:
        """Binary double (RDB_TYPE_ZSET_2, IEEE 754 little-endian)."""
        return struct.unpack("<d", self.read(8))[0]

    def read_uint16_le(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def read_uint32_le(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_uint64_le(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]


# ---------------------------------------------------------------------------
# Listpack decoder
# Encoding reference: Redis listpack.c
#   0xxxxxxx          7-bit uint         (0x00–0x7F)
#   10xxxxxx          6-bit string       (0x80–0xBF), len = low 6 bits
#   110xxxxx yyyyyyyy 13-bit signed int  (0xC0–0xDF)
#   1110xxxx xxxxxxxx 12-bit string      (0xE0–0xEF), len = low 4+8 bits
#   0xF1              16-bit signed int
#   0xF2              24-bit signed int
#   0xF3              32-bit signed int
#   0xF4              64-bit signed int
#   0xFF              end marker
# Each element is followed by a variable-length backlen field (bytes with
# bit-7 set are continuation; the final byte has bit-7 clear).
# ---------------------------------------------------------------------------
def _decode_listpack(data: bytes) -> List[bytes]:
    r = _Reader(data)
    _total_bytes = r.read_uint32_le()
    _num_elements = r.read_uint16_le()
    result = []
    while r.peek_byte() != 0xFF:
        result.append(_lp_read_element(r))
    return result


def _lp_read_element(r: _Reader) -> bytes:
    first = r.read_byte()

    if first & 0x80 == 0:
        # 7-bit uint: 0xxxxxxx
        _lp_skip_backlen(r)
        return str(first & 0x7F).encode()

    elif first & 0xC0 == 0x80:
        # 6-bit string: 10xxxxxx
        s = r.read(first & 0x3F)
        _lp_skip_backlen(r)
        return s

    elif first & 0xE0 == 0xC0:
        # 13-bit signed int: 110xxxxx yyyyyyyy
        val = ((first & 0x1F) << 8) | r.read_byte()
        if val >= 0x1000:
            val -= 0x2000
        _lp_skip_backlen(r)
        return str(val).encode()

    elif first & 0xF0 == 0xE0:
        # 12-bit string: 1110xxxx xxxxxxxx
        s = r.read(((first & 0x0F) << 8) | r.read_byte())
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
    """Skip the backlen field that follows each listpack element.

    The backlen encodes the total size of the preceding element (encoding +
    data bytes, NOT including the backlen itself).  Bytes with bit-7 set
    indicate continuation; the final byte has bit-7 clear.
    """
    b = r.read_byte()
    while b & 0x80:
        b = r.read_byte()


# ---------------------------------------------------------------------------
# Ziplist decoder (legacy types: LIST_ZIPLIST, HASH_ZIPLIST, ZSET_ZIPLIST)
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
    prevlen = r.read_byte()
    if prevlen == 0xFF:
        return None
    if prevlen == 0xFE:
        r.read(4)  # 5-byte prevlen: already read first byte, skip 4 more

    encoding = r.read_byte()
    enc_hi = encoding >> 6

    if enc_hi == 0:                 # 6-bit string
        return r.read(encoding & 0x3F)
    elif enc_hi == 1:               # 14-bit string
        return r.read(((encoding & 0x3F) << 8) | r.read_byte())
    elif enc_hi == 2:               # 32-bit string
        return r.read(struct.unpack(">I", r.read(4))[0])
    elif encoding == 0xC0:          # int16
        return str(struct.unpack("<h", r.read(2))[0]).encode()
    elif encoding == 0xD0:          # int32
        return str(struct.unpack("<i", r.read(4))[0]).encode()
    elif encoding == 0xE0:          # int64
        return str(struct.unpack("<q", r.read(8))[0]).encode()
    elif encoding == 0xF0:          # int24
        b = r.read(3)
        return str(struct.unpack("<i", b + b'\x00')[0]).encode()
    elif encoding == 0xFE:          # int8
        return str(struct.unpack("b", r.read(1))[0]).encode()
    elif 0xF1 <= encoding <= 0xFD:  # 4-bit uint (0–12)
        return str(encoding - 0xF1).encode()
    return None


# ---------------------------------------------------------------------------
# Intset decoder
# ---------------------------------------------------------------------------
def _decode_intset(data: bytes) -> List[bytes]:
    r = _Reader(data)
    encoding = r.read_uint32_le()           # 2, 4, or 8
    length   = r.read_uint32_le()
    fmt      = {2: "<h", 4: "<i", 8: "<q"}[encoding]
    return [str(struct.unpack(fmt, r.read(encoding))[0]).encode()
            for _ in range(length)]


# ---------------------------------------------------------------------------
# Type-specific value parsers
# ---------------------------------------------------------------------------
def _parse_string(r: _Reader) -> bytes:
    return r.read_string()


def _parse_list(r: _Reader) -> List[bytes]:
    n, _ = r.read_length()
    return [r.read_string() for _ in range(n)]


def _parse_set(r: _Reader) -> List[bytes]:
    n, _ = r.read_length()
    return [r.read_string() for _ in range(n)]


def _parse_zset(r: _Reader) -> List[Tuple[bytes, float]]:
    n, _ = r.read_length()
    return [(r.read_string(), r.read_double()) for _ in range(n)]


def _parse_zset2(r: _Reader) -> List[Tuple[bytes, float]]:
    n, _ = r.read_length()
    return [(r.read_string(), r.read_double_binary()) for _ in range(n)]


def _parse_hash(r: _Reader) -> Dict[bytes, bytes]:
    n, _ = r.read_length()
    return {r.read_string(): r.read_string() for _ in range(n)}


def _parse_list_ziplist(r: _Reader) -> List[bytes]:
    return _decode_ziplist(r.read_string())


def _parse_set_intset(r: _Reader) -> List[bytes]:
    return _decode_intset(r.read_string())


def _parse_zset_ziplist(r: _Reader) -> List[Tuple[bytes, float]]:
    elems = _decode_ziplist(r.read_string())
    return [(elems[i], float(elems[i + 1])) for i in range(0, len(elems), 2)]


def _parse_hash_ziplist(r: _Reader) -> Dict[bytes, bytes]:
    elems = _decode_ziplist(r.read_string())
    return {elems[i]: elems[i + 1] for i in range(0, len(elems), 2)}


def _parse_list_quicklist(r: _Reader) -> List[bytes]:
    n, _ = r.read_length()
    result = []
    for _ in range(n):
        result.extend(_decode_ziplist(r.read_string()))
    return result


def _parse_list_quicklist2(r: _Reader) -> List[bytes]:
    """Quicklist v2 (RDB v10): each node carries a container type.
    container=1  plain bytes node
    container=2  listpack node
    """
    n, _ = r.read_length()
    result = []
    for _ in range(n):
        data = r.read_string()
        container, _ = r.read_length()
        if container == 2:
            result.extend(_decode_listpack(data))
        else:
            result.append(data)
    return result


def _parse_hash_listpack(r: _Reader) -> Dict[bytes, bytes]:
    elems = _decode_listpack(r.read_string())
    return {elems[i]: elems[i + 1] for i in range(0, len(elems), 2)}


def _parse_zset_listpack(r: _Reader) -> List[Tuple[bytes, float]]:
    elems = _decode_listpack(r.read_string())
    return [(elems[i], float(elems[i + 1])) for i in range(0, len(elems), 2)]


# ---------------------------------------------------------------------------
# Stream parsers
# ---------------------------------------------------------------------------
def _read_stream_cgroups(r: _Reader, is_v2: bool) -> List[Dict]:
    """Read consumer groups shared by both stream type 15 and type 19."""
    num_cgroups, _ = r.read_length()
    cgroups = []
    for _ in range(num_cgroups):
        cg_name = r.read_string()

        # Consumer group last-delivered ID – both ms and seq are length-encoded
        cg_last_ms,  _ = r.read_length()
        cg_last_seq, _ = r.read_length()

        # entries_read only present in STREAM_LISTPACKS_2 (type 19)
        entries_read = None
        if is_v2:
            entries_read, _ = r.read_length()

        # Global PEL
        num_pel, _ = r.read_length()
        pel = []
        for _ in range(num_pel):
            raw_id        = r.read(16)           # streamID: ms(8) + seq(8) raw LE
            delivery_time = r.read_uint64_le()   # rdbLoadMillisecondTime → raw LE
            delivery_count, _ = r.read_length()  # rdbLoadLen → length-encoded
            pel.append({
                "id": raw_id,
                "delivery_time": delivery_time,
                "delivery_count": delivery_count,
            })

        # Consumers
        num_consumers, _ = r.read_length()
        consumers = []
        for _ in range(num_consumers):
            name      = r.read_string()
            seen_time = r.read_uint64_le()       # rdbLoadMillisecondTime → raw LE
            # active_time only in STREAM_LISTPACKS_3 (type 21, RDB v11) – not v10
            cpel_count, _ = r.read_length()
            cpel_ids = [r.read(16) for _ in range(cpel_count)]
            consumers.append({
                "name":      name,
                "seen_time": seen_time,
                "pel_ids":   cpel_ids,
            })

        cgroups.append({
            "name":         cg_name,
            "last_id":      (cg_last_ms, cg_last_seq),
            "entries_read": entries_read,
            "pel":          pel,
            "consumers":    consumers,
        })
    return cgroups


def _parse_stream_listpacks(r: _Reader) -> Dict:
    """RDB_TYPE_STREAM_LISTPACKS (15) – Redis 5.0."""
    num_lp, _ = r.read_length()
    listpacks = []
    for _ in range(num_lp):
        master_id = r.read_string()   # raw streamID bytes (16 bytes as string)
        lp_data   = r.read_string()
        listpacks.append({"master_id": master_id, "data": lp_data})

    # All stream metadata is length-encoded (rdbSaveLen / rdbLoadLen)
    length,   _ = r.read_length()
    last_ms,  _ = r.read_length()
    last_seq, _ = r.read_length()
    # first_id / max_deleted_entry_id / entries_added only in type 19

    cgroups = _read_stream_cgroups(r, is_v2=False)

    return {
        "listpacks": listpacks,
        "length":    length,
        "last_id":   (last_ms, last_seq),
        "cgroups":   cgroups,
    }


def _parse_stream_listpacks2(r: _Reader) -> Dict:
    """RDB_TYPE_STREAM_LISTPACKS_2 (19) – Redis 7.0 (RDB v10).

    Extends type 15 with first_id, max_deleted_entry_id, entries_added,
    and per-consumer-group entries_read.
    """
    num_lp, _ = r.read_length()
    listpacks = []
    for _ in range(num_lp):
        master_id = r.read_string()
        lp_data   = r.read_string()
        listpacks.append({"master_id": master_id, "data": lp_data})

    length,       _ = r.read_length()
    last_ms,      _ = r.read_length()
    last_seq,     _ = r.read_length()
    first_ms,     _ = r.read_length()
    first_seq,    _ = r.read_length()
    max_del_ms,   _ = r.read_length()
    max_del_seq,  _ = r.read_length()
    entries_added, _ = r.read_length()

    cgroups = _read_stream_cgroups(r, is_v2=True)

    return {
        "listpacks":               listpacks,
        "length":                  length,
        "last_id":                 (last_ms, last_seq),
        "first_id":                (first_ms, first_seq),
        "max_deleted_entry_id":    (max_del_ms, max_del_seq),
        "entries_added":           entries_added,
        "cgroups":                 cgroups,
    }


# ---------------------------------------------------------------------------
# Main type dispatch table
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
    RDB_TYPE_STREAM_LISTPACKS:   _parse_stream_listpacks,
    RDB_TYPE_STREAM_LISTPACKS_2: _parse_stream_listpacks2,
}


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------
def parse_rdb(data: bytes) -> RDBFile:
    """Parse a complete RDB dump from *data* and return an :class:`RDBFile`."""
    r = _Reader(data)

    magic = r.read(5)
    if magic != b"REDIS":
        raise ValueError(f"Not an RDB file (bad magic: {magic!r})")

    version = int(r.read(4))
    if version > 10:
        raise ValueError(f"Unsupported RDB version: {version} (max supported: 10)")

    rdb = RDBFile(version=version)
    current_db = 0
    expire_ms: Optional[int] = None
    lru_idle:  Optional[int] = None
    lfu_freq:  Optional[int] = None

    while True:
        opcode = r.read_byte()

        if opcode == RDB_OPCODE_EOF:
            if r.remaining() >= 8:
                _checksum = r.read(8)   # CRC64; verification left to caller
            break

        elif opcode == RDB_OPCODE_AUX:
            k = r.read_string()
            v = r.read_string()
            rdb.aux[k] = v

        elif opcode == RDB_OPCODE_RESIZEDB:
            r.read_length()   # db_size
            r.read_length()   # expires_size

        elif opcode == RDB_OPCODE_SELECTDB:
            current_db, _ = r.read_length()

        elif opcode == RDB_OPCODE_EXPIRETIME_MS:
            expire_ms = struct.unpack("<Q", r.read(8))[0]

        elif opcode == RDB_OPCODE_EXPIRETIME:
            expire_ms = struct.unpack("<I", r.read(4))[0] * 1000

        elif opcode == RDB_OPCODE_IDLE:
            # LRU idle time (seconds) for the immediately following key
            lru_idle, _ = r.read_length()

        elif opcode == RDB_OPCODE_FREQ:
            # LFU frequency byte for the immediately following key
            lfu_freq = r.read_byte()

        elif opcode in (RDB_OPCODE_FUNCTION2, RDB_OPCODE_FUNCTION):
            # Serialised Lua function library – store the payload blob
            rdb.functions.append(r.read_string())

        elif opcode == RDB_OPCODE_MODULE_AUX:
            # Module auxiliary data – module ID + opaque payload.
            # Without the module's own type handler we cannot safely skip
            # an arbitrary amount of data, so we raise rather than silently
            # mis-parse the rest of the file.
            raise ValueError(
                "RDB_OPCODE_MODULE_AUX encountered: module auxiliary data "
                "cannot be parsed without the module's type handler"
            )

        else:
            # opcode is the value-type byte
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
                lru_idle=lru_idle,
                lfu_freq=lfu_freq,
            ))
            expire_ms = None
            lru_idle  = None
            lfu_freq  = None

    return rdb


def parse_rdb_file(path: str) -> RDBFile:
    """Parse an RDB file from disk."""
    with open(path, "rb") as f:
        return parse_rdb(f.read())


def iter_entries(path: str) -> Iterator[RDBEntry]:
    """Iterate over all :class:`RDBEntry` objects in an RDB file."""
    yield from parse_rdb_file(path).entries
