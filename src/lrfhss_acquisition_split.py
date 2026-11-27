"""Opt-in two-ID acquisition with bounded, block-ordered result merging.

Each existing AcquisitionProcess searches one ID's disjoint header channels.
Numerics and validation are unchanged. The owner sees the original global
channel traversal only after both workers complete the same input block.
"""
from collections import Counter, deque
import time
from lrfhss_acquisition_process import AcquisitionProcess


def partition_channels(nominal, options):
    aliases = options.get('channel_map')
    if aliases is None or len(aliases) != len(nominal):
        raise ValueError('Split acquisition requires explicit channel aliases')
    if options.get('channelizer', 'per_channel') != 'per_channel' or options.get('carrier_search_hz', 4500.) != 4500.:
        raise ValueError('Split acquisition currently supports the narrow per-channel search only')
    groups = {}
    for index, channel in enumerate(aliases):
        ids = {hop_id for hop_id, _ in channel}
        if len(ids) != 1:
            raise ValueError('Split acquisition requires disjoint ID header channels')
        groups.setdefault(next(iter(ids)), []).append(index)
    if len(groups) != 2:
        raise ValueError('Split acquisition requires exactly two hop IDs')
    return tuple(tuple(indices) for indices in groups.values())


def merge_results(results, partitions, channel_count):
    coordinates = [(r['block_id'], r['start'], r['end']) for r in results]
    if len(results) != 2 or coordinates[0] != coordinates[1]:
        raise RuntimeError('Split acquisition result coordinates disagree')
    noise = [None] * channel_count
    references = [None] * channel_count
    offers = []
    counts = Counter()
    processing = Counter()
    for result, indices in zip(results, partitions):
        if len(result['noise']) != len(indices) or len(result['energy_references']) != len(indices):
            raise RuntimeError('Split acquisition channel state length changed')
        for local, global_index in enumerate(indices):
            noise[global_index] = result['noise'][local]
            references[global_index] = result['energy_references'][local]
        for fields in result['offers']:
            local = fields[1]
            if not isinstance(local, int) or not 0 <= local < len(indices):
                raise RuntimeError('Split acquisition returned an invalid channel index')
            offers.append((fields[0], indices[local], *fields[2:]))
        counts.update(result['counts'])
        for key, value in result['processing'].items():
            if key.endswith('_max_s'):
                processing[key] = max(processing[key], value)
            else:
                processing[key] += value
    # Stable within-channel order retains CFO rank and local suppression ties.
    offers.sort(key=lambda fields: fields[1])
    completed = max(r['completed_wall'] for r in results)
    began = min(r['completed_wall'] - r['wall_s'] for r in results)
    block_id, start, end = coordinates[0]
    return dict(kind='block', block_id=block_id, start=start, end=end,
                offers=offers, noise=noise, energy_references=references,
                counts=dict(counts), processing=dict(processing),
                cpu_s=sum(r['cpu_s'] for r in results), wall_s=completed-began,
                completed_wall=completed)


class _ReadyView:
    def __init__(self, owner):
        self.owner = owner

    def empty(self):
        owner = self.owner
        return owner.closed or not (all(owner.results) or any(not child.outbox.empty() for child in owner.children))


class SplitAcquisitionProcess:
    def __init__(self, fs, spb, nominal, trigger, options, *, capacity=8, timeout=30.):
        self.partitions = partition_channels(nominal, options)
        self.capacity, self.timeout, self.fs = capacity, timeout, fs
        self.channel_count = len(nominal)
        self.children = []
        self.results = [deque(), deque()]
        self.pending = deque()
        self.counters = Counter()
        self.next_block = self.watermark = self.submitted_sample = 0
        self.closed = False
        self.outbox = _ReadyView(self)
        started = time.perf_counter()
        try:
            for indices in self.partitions:
                subset = dict(options, channel_map=tuple(options['channel_map'][i] for i in indices))
                self.children.append(AcquisitionProcess(fs, spb, tuple(nominal[i] for i in indices),
                    trigger, subset, capacity=capacity, timeout=timeout))
        except BaseException:
            self.close()
            raise
        self.ready = dict(kind='ready', pid=self.children[0].ready['pid'],
                          pids=[child.ready['pid'] for child in self.children],
                          channel_partitions=self.partitions)
        self.counters['readiness_wall_s'] = time.perf_counter()-started

    @property
    def processes(self):
        return tuple(child.process for child in self.children)

    @property
    def process(self):
        # Compatibility for legacy single-PID inspection; complete PIDs are
        # exposed in processes and stats(), and must be used for CPU accounting.
        return self.children[0].process

    @property
    def outstanding(self):
        return len(self.pending)

    def is_alive(self):
        return not self.closed and all(child.is_alive() for child in self.children)

    def submit(self, raw, start):
        if self.closed or self.outstanding >= self.capacity:
            raise RuntimeError('Split acquisition admission bound exceeded')
        if start != self.submitted_sample or len(raw) % 2 or len(raw) > 100000:
            raise ValueError('Split acquisition input must be contiguous, at most 50 ms')
        for child in self.children:
            child.submit(raw, start)
        end = start + len(raw)//2
        self.pending.append((self.next_block, start, end, time.perf_counter()))
        self.next_block += 1
        self.submitted_sample = end
        self.counters['input_messages'] += 1
        self.counters['outstanding_high_watermark'] = max(self.counters['outstanding_high_watermark'], self.outstanding)
        self.counters['lag_high_watermark_samples'] = max(self.counters['lag_high_watermark_samples'], end-self.watermark)

    def collect(self, wait=False):
        if self.closed:
            return []
        collected = []
        while True:
            for child, buffered in zip(self.children, self.results):
                buffered.extend(child.collect())
            while self.pending and all(self.results):
                result = merge_results([buffered.popleft() for buffered in self.results],
                                       self.partitions, self.channel_count)
                block_id, start, end, submitted = self.pending.popleft()
                if (result['block_id'], result['start'], result['end']) != (block_id, start, end) or start != self.watermark:
                    raise RuntimeError('Split acquisition output order corrupted')
                self.watermark = end
                self.counters['output_messages'] += 1
                self.counters['worker_cpu_s'] += result['cpu_s']
                self.counters['worker_wall_s'] += result['wall_s']
                self.counters['result_latency_max_s'] = max(self.counters['result_latency_max_s'], time.perf_counter()-submitted)
                self.counters['offers'] += len(result['offers'])
                collected.append(result)
            if collected or not wait or not self.pending:
                return collected
            # Block on one missing half rather than spinning in the owner.
            index = next(i for i, buffered in enumerate(self.results) if not buffered)
            self.results[index].extend(self.children[index].collect(wait=True))

    def stats(self):
        children = [child.stats() for child in self.children]
        transport = {key:sum(row.get(key, 0) for row in children) for key in
                     ('input_bytes','output_bytes','queue_send_wall_s','pipe_send_wall_s','pipe_send_messages','pipe_send_bytes')}
        return dict(self.counters, **transport, capacity=self.capacity, input_capacity=self.capacity,
                    output_capacity=self.capacity, outstanding=self.outstanding,
                    input_queue_depth=max((row['input_queue_depth'] for row in children), default=0),
                    output_queue_depth=max((row['output_queue_depth']+len(buffered) for row,buffered in zip(children,self.results)), default=0),
                    processed_through_sample=self.watermark, submitted_sample=self.submitted_sample,
                    lag_samples=self.submitted_sample-self.watermark, ready=getattr(self,'ready',None),
                    closed=self.closed, pid=children[0]['pid'] if children else None,
                    pids=[row['pid'] for row in children], children=children,
                    worker_failures=sum(row.get('worker_failures',0) for row in children),
                    progress_timeouts=sum(row.get('progress_timeouts',0) for row in children),
                    process_count=len(children), channel_partitions=self.partitions)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.counters['cancelled_blocks'] += self.outstanding
        self.pending.clear()
        errors = []
        for child in self.children:
            try:
                child.close()
            except BaseException as error:
                errors.append(repr(error))
        for buffered in self.results:
            buffered.clear()
        if errors:
            raise RuntimeError('Split acquisition shutdown failures: ' + '; '.join(errors))
