"""Robust background-power estimates from decimated channel spectra.

Each Hann periodogram bin of stationary complex Gaussian noise is exponential.
Dividing its q-quantile by -log(1-q) estimates its mean. Channelizer/decimation
response is removed before pooling bins; the result is converted back to the
wide channel's mean-square power. Narrow signals contaminate a minority of bins
instead of the entire block mean. There is no activity-dependent update veto.
"""
import math
import numpy as np
from scipy import signal


class SpectralNoise:
    def __init__(self, fs, decimation, sos, *, quantile=.2, tau_s=.35, guard=False):
        if not 0 < quantile < 1 or tau_s <= 0:
            raise ValueError('Invalid noise estimator parameters')
        self.fs, self.decimation = fs, decimation
        self.quantile, self.tau_s = quantile, tau_s
        self.nfft = 512 if guard else 256
        self.window = np.hanning(self.nfft)
        self.window_energy = float(self.window @ self.window)
        low_fs = fs/decimation
        frequencies = np.fft.fftfreq(self.nfft, 1/low_fs)
        self.bins = ((np.abs(frequencies) >= 5500) & (np.abs(frequencies) <= 7300)
                     if guard else np.abs(frequencies) <= 4500)
        # Alias power, including stopband leakage, for white input noise.
        alias = np.zeros(self.nfft)
        for k in range(-decimation//2, decimation//2):
            _, h = signal.sosfreqz(sos, worN=2*np.pi*(frequencies+k*low_fs)/fs)
            alias += np.abs(h)**2
        self.response = alias[self.bins]
        impulse = np.zeros(8192); impulse[0] = 1
        self.channel_gain = float(np.sum(signal.sosfilt(sos, impulse)**2))
        self.scale = decimation*self.channel_gain / -math.log1p(-quantile)
        self.value = None
        self.observation = None

    def update(self, low, duration_s):
        low = np.asarray(low)
        if len(low) < self.nfft:
            return self.value
        windows = np.lib.stride_tricks.sliding_window_view(low, self.nfft)[::self.nfft//2]
        spectrum = np.abs(np.fft.fft(windows*self.window, axis=1))**2/self.window_energy
        observation = float(np.quantile(spectrum[:, self.bins]/self.response, self.quantile))*self.scale
        self.observation = max(observation, 1e-16)
        alpha = -math.expm1(-duration_s/self.tau_s)
        self.value = self.observation if self.value is None else max(
            (1-alpha)*self.value + alpha*self.observation, 1e-16)
        return self.value
