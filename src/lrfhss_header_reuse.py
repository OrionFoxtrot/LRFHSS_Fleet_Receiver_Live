"""Exact immutable PHDR result reuse across separately scheduled retries.

The selection identity hashes ALL IQ bytes and includes base, dtype, shape,
candidate fields, sample rate and FFT backend. Add selected frequency and
validator backend. No nearby-frequency or nearby-time matching is permitted.
Only one small prior result is retained per bounded candidate; IQ is not cached.
"""
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class ValidationResult:
    identity: tuple
    header: object
    detail: dict


def validate(validator, z, base, candidate, frequency, selection_identity,
             batch_header, previous=None):
    key = ('physical-validator-v1', selection_identity, frequency, bool(batch_header))
    if previous is not None and previous.identity == key:
        return deepcopy(previous.header), deepcopy(previous.detail), previous, True
    header, detail = validator(z, base, candidate, frequency)
    result = ValidationResult(key, deepcopy(header), deepcopy(detail))
    return header, detail, result, False
