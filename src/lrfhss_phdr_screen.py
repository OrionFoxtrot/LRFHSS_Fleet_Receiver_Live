"""Conservative low-rate PHDR correlation screen over one immutable IQ window.

The screen shares a padded full-window FFT between nearby frequency hypotheses.
It applies the EXISTING sixth-order Butterworth forward/backward magnitude in
the frequency domain, evaluates phase at the original sample coordinates, and
can only decline a clearly implausible coarse-sync hypothesis. A pass/uncertain
result always requests the unchanged full-rate timing, FEC, and CRC validator.

This is an approximate pre-validator, not an exact replacement for sosfiltfilt:
zero padding changes endpoint transients; FFT arithmetic and interpolation add
small error. Guards below deliberately fail open, including a 0.10 correlation
margin below the receiver's 0.45 threshold. Release regression, and an option
to disable the screen, remain mandatory. No transmitter schedule/identity is
used. The context must not outlive the immutable candidate IQ transaction.
"""
from collections import Counter

import numpy as np
from scipy import fft
from lrfhss_metrics import measure as _measure


FS = 1_000_000
SUPPORTED_FACTORS = (16, 32, 64, 128, 256, 512)


class _SparseBaseband:
    """Four-point interpolation at original-rate coordinates, without rounding."""
    def __init__(self, values, factor, length):
        self.values, self.factor, self.length = values, factor, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        coordinate = np.asarray(index) / self.factor
        integral = np.floor(coordinate).astype(np.intp)
        t = coordinate - integral
        wm = -t * (t - 1) * (t - 2) / 6
        w0 = (t + 1) * (t - 1) * (t - 2) / 2
        w1 = -(t + 1) * t * (t - 2) / 2
        w2 = (t + 1) * t * (t - 1) / 6
        n = len(self.values)
        return (wm * self.values[(integral - 1) % n]
                + w0 * self.values[integral % n]
                + w1 * self.values[(integral + 1) % n]
                + w2 * self.values[(integral + 2) % n])


class ScreenContext:
    """One transaction's byte-bounded reusable spectral frontend.

    ``screen`` returns ``reject=False`` for all unsupported/uncertain inputs.
    ``decimation=64`` gives 15,625 samples/s, with exact 1 MHz timing queries.
    The factor argument supports explicitly requested experiments only; the
    release profile is responsible for choosing its validated factor.
    """
    def __init__(self, z, counters=None, decimation=64):
        self.z = z
        self.counters = Counter() if counters is None else counters
        self.factor = int(decimation)
        if self.factor not in SUPPORTED_FACTORS:
            raise ValueError('Unsupported PHDR screen decimation factor')
        self.spectrum = None
        self.m = None
        self.length = len(z)
        # 20 ms separates the periodically extended FFT's input endpoints.
        # Interior safety checks below also keep requested samples >=30 ms
        # from either endpoint of the ORIGINAL candidate window.
        self.low_length = fft.next_fast_len((self.length + 20_000
                                            + self.factor - 1) // self.factor)
        self.fft_length = self.low_length * self.factor
        self.input_peak_bound = None

    def _frontend(self):
        counts = self.counters
        if self.spectrum is not None:
            counts['phdr_frontend_reuse_hits'] += 1
            return
        counts['phdr_frontend_reuse_misses'] += 1
        counts['phdr_frontend_computations'] += 1
        counts['phdr_frontend_fullrate_samples'] += self.length
        counts['phdr_frontend_fft_samples'] += self.fft_length
        with _measure(counts, 'phdr_kernel_frontend_fft'):
            self.spectrum = fft.fft(np.asarray(self.z, dtype=np.complex64),
                                    self.fft_length, workers=1)
            self.m = np.rint(fft.fftfreq(self.low_length)
                             * self.low_length).astype(np.intp)
            self.input_peak_bound = float(np.max(abs(self.z.real))
                                          + np.max(abs(self.z.imag)))
        cache_bytes = self.spectrum.nbytes + self.m.nbytes
        counts['phdr_frontend_cache_peak_bytes'] = max(
            counts.get('phdr_frontend_cache_peak_bytes', 0), cache_bytes)
        counts['phdr_frontend_cache_current_bytes'] = cache_bytes

    def _baseband(self, frequency, cutoff):
        self._frontend()
        counts = self.counters
        with _measure(counts, 'phdr_kernel_frontend_lowrate_filter'):
            with _measure(counts, 'phdr_kernel_frontend_translation'):
                spacing = FS / self.fft_length
                coarse_bin = round(frequency / spacing)
                residual = frequency - coarse_bin * spacing
                frequencies = self.m * spacing - residual
            # For Butterworth order6, |H|² = 1/(1+Omega**12), where Omega
            # includes the same bilinear prewarping as scipy.signal.butter.
            # sosfiltfilt's zero-phase AMPLITUDE is |H|², not |H| or |H|^4.
            with _measure(counts, 'phdr_kernel_frontend_response'):
                ratio = np.tan(np.pi * frequencies / FS) / np.tan(np.pi * cutoff / FS)
                magnitude = 1 / (1 + ratio ** 12)
            with _measure(counts, 'phdr_kernel_frontend_spectrum_copy_multiply'):
                narrowed = self.spectrum[(coarse_bin + self.m) % self.fft_length] * magnitude
            with _measure(counts, 'phdr_kernel_frontend_decimation'):
                low = fft.ifft(narrowed, workers=1) / self.factor
        with _measure(counts, 'phdr_kernel_frontend_residual_cfo'):
            low *= np.exp((-2j * np.pi * residual * self.factor / FS)
                          * np.arange(self.low_length))
        counts['phdr_frontend_lowrate_samples'] += self.low_length
        counts['phdr_frontend_hypotheses_served'] += 1
        counts['phdr_frontend_decimation_' + str(self.factor)] += 1
        counts['phdr_residual_cfo_operations'] += 1
        return _SparseBaseband(low, self.factor, self.length)

    def screen(self, frequency, cutoff, anchor):
        """Test all seven original coarse timing trials; fail open on doubt."""
        import lrfhss_live as phy
        counts = self.counters
        counts['phdr_screen_attempts'] += 1

        def decision(reject, reason, **details):
            counts['phdr_screen_rejected' if reject else 'phdr_screen_passed'] += 1
            if reason not in ('low_sync', 'sync_possible'):
                counts['phdr_screen_uncertain'] += 1
                counts['phdr_screen_guard_' + reason] += 1
            return dict(reject=bool(reject), reason=reason,
                        decimation=self.factor, threshold=.35, **details)

        if (phy.FS != FS or cutoff != 250. or not np.isfinite(frequency)
                or abs(frequency) >= FS / 2
                or not np.isfinite(anchor)):
            return decision(False, 'unsupported')
        deltas = np.asarray([0, -512, 512, -1024, 1024, -1536, 1536])
        centers = int(anchor) + deltas[:, None] + np.arange(113) * phy.SPB
        requested_min = int(np.min(centers) - phy.LOOK)
        requested_max = int(np.max(centers) + phy.LOOK)
        if requested_min < 30_000 or self.length - 1 - requested_max < 30_000:
            return decision(False, 'edge_window')
        with _measure(counts, 'phdr_kernel_screen'):
            baseband = self._baseband(float(frequency), float(cutoff))
            with _measure(counts, 'phdr_kernel_screen_phase_sampling'):
                left = baseband[centers - phy.LOOK]
                right = baseband[centers + phy.LOOK]
                values = np.concatenate((left.ravel(), right.ravel()))
                if not np.all(np.isfinite(values)):
                    return decision(False, 'nonfinite')
                # Avoid amplifying frontend approximation error through angle
                # close to a complex zero. Peak uses a conservative I+Q bound.
                minimum = float(np.min(abs(values)))
                if minimum <= max(1e-12, self.input_peak_bound * 1e-4):
                    return decision(False, 'low_amplitude', minimum_amplitude=minimum)
                difference = np.angle(right) - np.angle(left)
                wrapped = (difference + np.pi) % (2 * np.pi) - np.pi
                branch_distance = float(np.min(abs(abs(wrapped) - np.pi / 2)))
                if branch_distance < 1e-4:
                    return decision(False, 'phase_branch', branch_distance=branch_distance)
            with _measure(counts, 'phdr_kernel_screen_soft_sync'):
                softs = phy._soft_trials(phy._SampledPhase(baseband), centers)
                scores = [float(phy.sync_score(soft[41:73])) for soft in softs]
            if not np.all(np.isfinite(scores)):
                return decision(False, 'nonfinite')
            maximum = max(scores)
            return decision(maximum < .35,
                            'low_sync' if maximum < .35 else 'sync_possible',
                            maxscore=maximum, scores=scores,
                            minimum_amplitude=minimum,
                            branch_distance=branch_distance)

    def close(self):
        self.spectrum = self.m = self.z = None
        self.counters['phdr_frontend_cache_current_bytes'] = 0
