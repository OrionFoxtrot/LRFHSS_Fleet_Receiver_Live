"""Opt-in hop-ID configuration and explicit search-channel aliases.

An RF channel can carry different IDs and different PHDR copies. Channel
indices are never physical-copy indices except in the preserved default path.
"""
from numbers import Integral

DEFAULT_ALLOWED_HOP_IDS = (256,)


def normalize_hop_ids(values):
    """Canonical nonempty 9-bit ID set for a LiveDecoder/profile override."""
    if isinstance(values, (str, bytes)):
        raise ValueError('allowed_hop_ids must be a collection of integers')
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError('allowed_hop_ids must be a collection of integers') from exc
    if not values or any(isinstance(x, bool) or not isinstance(x, Integral) or
                         not 0 <= x <= 511 for x in values):
        raise ValueError('allowed_hop_ids must contain integers in 0..511')
    return tuple(sorted(set(int(x) for x in values)))


def parse_hop_ids(value):
    """CLI: comma-separated decimal/hexadecimal IDs, or all."""
    if value.strip().lower() == 'all':
        return tuple(range(512))
    try:
        return normalize_hop_ids(int(x.strip(), 16 if x.strip().lower().startswith('0x') else 10)
                                 for x in value.split(','))
    except (TypeError, ValueError) as exc:
        raise ValueError('use comma-separated IDs in 0..511 (hex accepted), or all') from exc


def header_search_plan(allowed_hop_ids, hop_frequencies, grid=1, bw_code=6):
    """Return unique RF channels and parallel tuples of (hop ID, copy index)."""
    ids = normalize_hop_ids(allowed_hop_ids)
    channels = {}
    for hop_id in ids:
        for copy, hz in enumerate(hop_frequencies(grid, bw_code, hop_id, 3, 0)):
            channels.setdefault(float(hz), []).append((hop_id, copy))
    return tuple(channels), tuple(tuple(aliases) for aliases in channels.values())
