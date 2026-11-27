"""Bounded owner-written header snapshots for spawn workers.

A slot is immutable from publish until the terminal worker message. Workers
drop active views BEFORE sending that message. Idle mappings are cached. The owner alone releases/reuses slots and
closes/unlinks them, after stopping every worker on cancellation. Generations
prevent a stale/double release from freeing a newer lease. No live ring view
crosses a process boundary; one copy fills each snapshot directly from chunks.
"""
from dataclasses import dataclass
from collections import OrderedDict
from multiprocessing.shared_memory import SharedMemory
import numpy as np


@dataclass(frozen=True)
class SharedIQ:
    name: str
    generation: int
    nbytes: int

    def __len__(self):
        return self.nbytes


class SharedSnapshots:
    def __init__(self, capacity, max_snapshot_bytes=390000*16):
        self.capacity = capacity
        self.max_snapshot_bytes = max_snapshot_bytes
        self.slots = []
        self.closed = False
        self.copies = self.copied_bytes = self.peak_bytes = self.peak_leases = 0

    def snapshot(self, ring, start, stop):
        if self.closed:
            raise RuntimeError('Shared snapshots closed')
        if not ring.chunks or stop <= start or start < ring.chunks[0][0] or stop > ring.end:
            return None
        parts = []; expected = stop; dtype = None
        for sample, data in reversed(ring.chunks):
            if sample >= stop: continue
            if sample+len(data) <= start: break
            a, b = max(start, sample), min(stop, sample+len(data))
            if b <= a: continue
            if b != expected:
                return None
            dtype = data.dtype if dtype is None else np.result_type(dtype, data.dtype)
            parts.append((a-start, data[a-sample:b-sample]))
            expected = a
            if a == start: break
        if expected != start:
            return None
        size = (stop-start)*dtype.itemsize
        if size > self.max_snapshot_bytes:
            raise ValueError('Header snapshot exceeds configured shared-memory bound')
        slot = next((s for s in self.slots if not s['leased'] and s['memory'].size >= size), None)
        if slot is None:
            if len(self.slots) >= self.capacity:
                raise RuntimeError('Shared snapshot lease accounting exceeded queue bound')
            # Stable names/sizes allow workers to retain bounded mappings.
            # No retired generations remain mapped after owner slot growth.
            memory = SharedMemory(create=True, size=self.max_snapshot_bytes)
            slot = dict(memory=memory, generation=0, leased=False)
            self.slots.append(slot)
        target = np.ndarray((stop-start,), dtype=dtype, buffer=slot['memory'].buf)
        for offset, data in parts:
            target[offset:offset+len(data)] = data
        del target
        slot['generation'] += 1; slot['leased'] = True
        self.copies += 1; self.copied_bytes += size
        self.peak_bytes = max(self.peak_bytes, sum(s['memory'].size for s in self.slots))
        self.peak_leases = max(self.peak_leases, sum(s['leased'] for s in self.slots))
        return SharedIQ(slot['memory'].name, slot['generation'], size), dtype.str

    def release(self, descriptor):
        slot = next((s for s in self.slots if s['memory'].name == descriptor.name), None)
        if slot is None or not slot['leased'] or slot['generation'] != descriptor.generation:
            raise RuntimeError('Stale or duplicate shared IQ release')
        slot['leased'] = False

    def close(self):
        if self.closed: return
        # Called only after pool.close has joined/stopped all worker processes.
        for slot in self.slots:
            slot['memory'].close()
            slot['memory'].unlink()
        self.slots.clear(); self.closed = True

    def stats(self):
        return dict(capacity=self.capacity, max_snapshot_bytes=self.max_snapshot_bytes,
            allocated_bytes=sum(s['memory'].size for s in self.slots),
            leased=sum(s['leased'] for s in self.slots), slots=len(self.slots),
            peak_bytes=self.peak_bytes, peak_leases=self.peak_leases,
            copies=self.copies, copied_bytes=self.copied_bytes, closed=self.closed)


def attach(descriptor, dtype):
    """Worker owns returned handle; discard array then close before replying."""
    memory = SharedMemory(name=descriptor.name)
    try:
        item = np.dtype(dtype)
        if descriptor.nbytes > memory.size or descriptor.nbytes % item.itemsize:
            raise ValueError('Invalid shared IQ descriptor')
        data = np.ndarray((descriptor.nbytes//item.itemsize,), dtype=item, buffer=memory.buf)
        data.flags.writeable = False
        return memory, data
    except BaseException:
        memory.close()
        raise


class SharedReaders:
    """Worker-local mapping cache; no IQ result or active array is retained.

    Keeping mappings avoids re-faulting every page of a 2.6 MB window on every
    attempt. A retained mapping is NOT an active reader: only the running job
    may access its view. Owner reuse still waits for the terminal message.
    """
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.maps = OrderedDict()

    def array(self, descriptor, dtype):
        memory = self.maps.pop(descriptor.name, None)
        if memory is None:
            if len(self.maps) >= self.capacity:
                _, old = self.maps.popitem(last=False); old.close()
            memory = SharedMemory(name=descriptor.name)
        self.maps[descriptor.name] = memory
        item = np.dtype(dtype)
        if descriptor.nbytes > memory.size or descriptor.nbytes % item.itemsize:
            raise ValueError('Invalid shared IQ descriptor')
        data = np.ndarray((descriptor.nbytes//item.itemsize,), dtype=item, buffer=memory.buf)
        data.flags.writeable = False
        return data

    def close(self):
        for memory in self.maps.values(): memory.close()
        self.maps.clear()
