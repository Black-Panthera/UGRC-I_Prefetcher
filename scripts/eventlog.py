"""Readers for the ChampSim L2C event logs written with CHAMPSIM_EVENT_LOG=<prefix>.

<prefix>.acc.bin : one AccessRecord per L1D demand access reaching the L2C (detailed phase only)
<prefix>.pf.bin  : one PrefetchRecord per prefetch request outcome / unused eviction
Either file may be zstd-compressed (<name>.zst).
"""

import os
import subprocess

import numpy as np

ACCESS_DTYPE = np.dtype([
    ('instr_id', '<u8'), ('pline', '<u8'), ('vline', '<u8'), ('trigger_instr', '<u8'),
    ('timing', '<i4'), ('outcome', 'u1'), ('type', 'u1'), ('seen', 'u1'), ('flags', 'u1'),
])
PREFETCH_DTYPE = np.dtype([
    ('trigger_instr', '<u8'), ('target_pline', '<u8'), ('cycle', '<u4'), ('delta', '<i4'),
    ('event', 'u1'), ('fill_this_level', 'u1'), ('flags', 'u1'), ('pad', 'u1'), ('aux', '<u4'),
])
assert ACCESS_DTYPE.itemsize == 40 and PREFETCH_DTYPE.itemsize == 32

ACCESS_OUTCOMES = ['HIT', 'PF_HIT_TIMELY', 'PF_HIT_REUSE', 'UPSTREAM_PF_HIT', 'PF_HIT_LATE', 'MISS', 'MISS_MERGED']
PREFETCH_EVENTS = ['REJECTED_PQ_FULL', 'FILTERED_RESIDENT', 'MERGED_INFLIGHT_PF', 'MERGED_INFLIGHT_DEMAND',
                   'DROPPED_MSHR_FULL', 'ISSUED_FILL_L2', 'ISSUED_LLC_ONLY', 'EVICTED_UNUSED']
NO_TRIGGER = np.uint64(0xFFFFFFFFFFFFFFFF)
HEADER_SIZE = 16


def _read_bytes(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    if os.path.exists(path + '.zst'):
        return subprocess.run(['zstd', '-dc', path + '.zst'], check=True, capture_output=True).stdout
    raise FileNotFoundError(path)


def _load(path, dtype, magic):
    raw = _read_bytes(path)
    assert raw[:8] == magic, f'{path}: bad magic {raw[:8]!r}'
    rec_size = int.from_bytes(raw[8:12], 'little')
    assert rec_size == dtype.itemsize, f'{path}: record size {rec_size} != {dtype.itemsize}'
    body = raw[HEADER_SIZE:]
    assert len(body) % rec_size == 0, f'{path}: truncated ({len(body)} bytes)'
    return np.frombuffer(body, dtype=dtype)


def load_access(prefix):
    return _load(prefix + '.acc.bin', ACCESS_DTYPE, b'CSACCLOG')


def load_prefetch(prefix):
    return _load(prefix + '.pf.bin', PREFETCH_DTYPE, b'CSPFLOG\0')
