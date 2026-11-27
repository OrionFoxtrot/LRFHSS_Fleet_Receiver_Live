"""Configuration values shared by the LR-FHSS Fleet Manager GUI."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil

from lrfhss_live import (
    DEFAULT_CENTER_HZ, DEFAULT_GAIN_DB, DEFAULT_TRIGGER_DB,
    DEFAULT_CARRIER_SEARCH_HZ, BATCH10_FRONTEND_PROFILE, FRONTEND_SIC_PRESSURE_MODE,
    NUC_RR2_20S_V2_PROFILE, hop_frequencies,
)
from lrfhss_acquisition_split import partition_channels
from lrfhss_hop_ids import header_search_plan, normalize_hop_ids

# The GUI runs the Batch 10 front-end profile (adopted 2026-10-01): the Batch 9.10
# receiver path (screened PHDRs, u8 payload transport, grouped timing trials,
# deferred header-replica skip) on 7 PHDR / 4 payload workers, the 10.2 bounded
# payload retry, and PHDR interference cancellation (SIC) on one low-priority
# worker with the 10.4h proportional duty. Every packet the receiver decodes
# without SIC is decoded identically; SIC adds packets.
GUI_PROFILE = BATCH10_FRONTEND_PROFILE
HOP_SHARED = "single"
HOP_RR2 = "two"
LEGACY_HOP_MODES = {"shared_256": HOP_SHARED, "rr2_256_299": HOP_RR2}
RR2_TRIGGER_DB = NUC_RR2_20S_V2_PROFILE["trigger_db"]

APP_NAME = "DeSCENT LR-FHSS Fleet Manager"
HISTORY_POINTS = 300
STALE_SECONDS = 60.0  # Two nominal 30-second packet periods; display only.
LOST_SECONDS = 120.0
MAX_PACKET_RECORDS = 2_000
PACKET_TRIM_COUNT = 2_000
MAX_PACKET_TABLE_ROWS = 2_000

# The Windows helper sits beside the station DLLs. Source-only installations
# can still use their existing rtl_tcp from PATH.
FLEET_RTL_TCP = Path(__file__).resolve().parents[2] / "rtl_tcp_fleet.exe"
LIVE_USB_BYTES = 1024 * 1024
SHORT_USB_BYTES = 256 * 1024


def default_rtl_tcp() -> str:
    override = os.environ.get("RTL_TCP_EXE")
    if override:
        return override
    if os.name != "nt":
        return "rtl_tcp"
    dlls = ("librtlsdr.dll", "libusb-1.0.dll", "libwinpthread-1.dll")
    if FLEET_RTL_TCP.is_file() and all((FLEET_RTL_TCP.parent / name).is_file() for name in dlls):
        return str(FLEET_RTL_TCP)
    return "rtl_tcp.exe"


def rtl_tcp_fallbacks() -> list[str]:
    """Installed Windows helpers, keeping each executable with its own DLLs."""
    if os.name != "nt":
        return []
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library/bin/rtl_tcp.exe")
    candidates.extend((
        Path(os.environ.get("ProgramData", "C:/ProgramData")) / "radioconda/Library/bin/rtl_tcp.exe",
        Path.home() / "radioconda/Library/bin/rtl_tcp.exe",
    ))
    on_path = shutil.which("rtl_tcp.exe")
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(FLEET_RTL_TCP.with_name("rtl_tcp.exe"))
    return list(dict.fromkeys(str(path.resolve()) for path in candidates if path.is_file()))


@dataclass(slots=True)
class ReceiverConfig:
    mode: str
    hop_id_mode: str = HOP_SHARED
    rtl_tcp: str = field(default_factory=default_rtl_tcp)
    freq: int = DEFAULT_CENTER_HZ
    gain: float = DEFAULT_GAIN_DB
    ppm: int = 0
    port: int = 1234
    trigger_db: float | None = None
    carrier_search_hz: float = DEFAULT_CARRIER_SEARCH_HZ
    file_path: str = ""
    exhaustive_file: bool = False
    # Live mode: write the raw IQ to logs/iq/ for later re-decoding.
    record_iq: bool = False
    # One-shot live calibration; no packet decoder or recorder is started.
    autoset_only: bool = False

    payload_workers: int = GUI_PROFILE["payload_workers"]
    payload_queue_capacity: int = GUI_PROFILE["payload_queue_capacity"]
    header_workers: int = GUI_PROFILE["header_workers"]
    header_queue_capacity: int = GUI_PROFILE["header_queue_capacity"]
    header_candidate_capacity: int = GUI_PROFILE["header_candidate_capacity"]
    phdr_frontend: str = GUI_PROFILE["phdr_frontend"]
    phdr_decimation: int = GUI_PROFILE["phdr_decimation"]
    payload_transport: str = GUI_PROFILE["payload_transport"]
    dsp_soft_trials: str = GUI_PROFILE["dsp_soft_trials"]
    header_replica_skip: str = GUI_PROFILE.get("header_replica_skip", "off")
    payload_retry: str | None = None
    phdr_sic: str = GUI_PROFILE.get("phdr_sic", "off")
    sic_workers: int = GUI_PROFILE.get("sic_workers", 1)
    sic_payload_allowance: int = GUI_PROFILE.get("sic_payload_allowance", 1)
    usb_transfer_bytes: int = LIVE_USB_BYTES
    weak_signal_search: bool = False
    allowed_hop_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        self.hop_id_mode = LEGACY_HOP_MODES.get(self.hop_id_mode, self.hop_id_mode)
        if self.hop_id_mode not in (HOP_SHARED, HOP_RR2):
            raise ValueError("Unknown hop-ID mode")
        two = self.hop_id_mode == HOP_RR2
        ids = self.allowed_hop_ids
        if ids is None:
            ids = (256, 299) if two else (256,)
        self.allowed_hop_ids = normalize_hop_ids(ids)
        if len(ids) != (2 if two else 1):
            raise ValueError("Choose two hop IDs." if two else "Choose one hop ID.")
        if two and ids[0] == ids[1]:
            raise ValueError(f"Cannot start with hop IDs {ids[0]} and {ids[1]}. Choose two different IDs.")
        if self.allowed_hop_ids != (256,):
            self.phdr_sic = "off"
        if self.autoset_only and (self.allowed_hop_ids != (256,) or self.mode != "live"):
            raise ValueError("Auto Set requires live input and Single 256.")
        if self.trigger_db is None:
            self.trigger_db = RR2_TRIGGER_DB if two else DEFAULT_TRIGGER_DB
        if two:
            for name, value in NUC_RR2_20S_V2_PROFILE.items():
                if name in self.__dataclass_fields__ and name not in ("allowed_hop_ids", "trigger_db"):
                    setattr(self, name, value)
            self.carrier_search_hz = DEFAULT_CARRIER_SEARCH_HZ
            self.payload_retry = "adaptive" if self.weak_signal_search and self.mode == "live" else "off"
            hops, aliases = header_search_plan(self.allowed_hop_ids, hop_frequencies)
            try:
                partition_channels(hops, dict(decoder_options(self)["acquisition_options"], channel_map=aliases))
            except ValueError as exc:
                raise ValueError(f"Cannot split hop IDs {ids[0]} and {ids[1]}. {exc}.") from exc
        if self.payload_retry is None:
            self.payload_retry = "adaptive" if self.mode == "live" else GUI_PROFILE.get("payload_retry", "off")


def decoder_options(config: ReceiverConfig) -> dict:
    """LiveDecoder keyword arguments for the GUI (file and live modes share them)."""
    acquisition = {"carrier_search_hz": config.carrier_search_hz}
    if config.weak_signal_search:
        acquisition["gate_mode"] = "weak_signal"
    if config.hop_id_mode == HOP_RR2:
        options = dict(NUC_RR2_20S_V2_PROFILE)
        options.pop("trigger_db")
        options["allowed_hop_ids"] = config.allowed_hop_ids
        options["acquisition_options"] = dict(acquisition, noise_mode="guard20")
        if config.payload_retry != "off":
            options["payload_retry"] = config.payload_retry
        return options
    threaded = bool(config.header_workers)
    options = dict(
        payload_workers=config.payload_workers,
        payload_queue_capacity=config.payload_queue_capacity,
        header_workers=config.header_workers,
        header_queue_capacity=config.header_queue_capacity,
        header_candidate_capacity=config.header_candidate_capacity,
        header_validation_budget=None if threaded else 2,
        batch_header=threaded,
        header_grid_cache=threaded,
        acquisition_execution='process', acquisition_search_backend='batched',
        acquisition_options=acquisition,
        phdr_frequency_reuse=True, header_shared_iq=True,
        header_validation_reuse=True, header_idle_service=True, payload_fec_backend='numpy',
        candidate_transactions=threaded, header_oscillator_reuse=False,
        phdr_frontend=config.phdr_frontend if threaded else 'reference',
        phdr_decimation=config.phdr_decimation,
        payload_transport=config.payload_transport,
        dsp_soft_trials=config.dsp_soft_trials,
        # The skip needs the asynchronous PHDR pipeline; the synchronous
        # compatibility mode (0 header workers) keeps the reference behaviour.
        header_replica_skip=config.header_replica_skip if threaded else 'off',
        payload_retry=config.payload_retry,
        # SIC needs the asynchronous PHDR pipeline. Live: a short drain at stop;
        # files: every SIC tile is finished at the end of the recording.
        phdr_sic=config.phdr_sic if threaded else 'off',
        sic_workers=config.sic_workers,
        sic_pressure={"mode": FRONTEND_SIC_PRESSURE_MODE},
        sic_payload_allowance=config.sic_payload_allowance,
        sic_eof='flush' if config.mode == 'file' else 'skip')
    # Preserve the original constructor call for its default ID.
    if config.allowed_hop_ids != (256,):
        options["allowed_hop_ids"] = config.allowed_hop_ids
    return options
