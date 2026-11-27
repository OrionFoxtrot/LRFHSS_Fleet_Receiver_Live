"""Bounded timing distributions. Totals are exact; quantiles use the last 4096 calls."""
from collections import defaultdict, deque
from contextlib import contextmanager
from time import perf_counter
import numpy as np

class Timings:
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.values = defaultdict(lambda: deque(maxlen=capacity))
        self.counts = defaultdict(int)
        self.totals = defaultdict(float)
        self.maxima = defaultdict(float)

    def add(self, name, value):
        value = float(value)
        self.values[name].append(value)
        self.counts[name] += 1
        self.totals[name] += value
        self.maxima[name] = max(self.maxima[name], value)

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter()-start)

    def summary(self):
        return {k: dict(count=self.counts[k], total_s=self.totals[k],
                       mean_s=self.totals[k]/self.counts[k], max_s=self.maxima[k],
                       median_s=float(np.median(v)), p95_s=float(np.percentile(v,95)),
                       p99_s=float(np.percentile(v,99)), quantile_samples=len(v),
                       quantile_basis="all" if len(v)==self.counts[k] else "last_4096")
                for k,v in self.values.items() if v}
