"""GUI-neutral fleet/session state.

The window should render state, not decide what a packet means.  This module
owns session counters, per-ChipSat state, telemetry history, and CSV flattening.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
from typing import Any

from .config import HISTORY_POINTS, MAX_PACKET_RECORDS, STALE_SECONDS


@dataclass(slots=True)
class ChipSatState:
    """Latest state retained for one ChipSat."""

    csid: int
    first_seen: float
    last_seen: float
    count: int = 0
    latest: dict[str, Any] | None = None
    envelope: dict[str, Any] | None = None
    last_packet_counter: int | None = None
    estimated_counter_gaps: int = 0
    duplicate_counters: int = 0
    counter_discontinuities: int = 0
    recent_arrivals: deque[float] = field(default_factory=lambda: deque(maxlen=1024))
    recent_time_basis: str | None = None
    latest_phy_quality: dict[str, Any] | None = None
    out_of_order_results: int = 0
    _latest_source_position: tuple | None = None
    _counter_high_water: int | None = None
    _pending_gaps: set[int] = field(default_factory=set)

    def update(self, decoded: dict[str, Any], envelope: dict[str, Any], timestamp: float) -> None:
        self.count += 1
        self.last_seen = timestamp
        phy = envelope.get("phy", {})
        position = ((phy.get("receiver_id"), phy.get("packet_start_sample"))
                    if phy.get("receiver_id") is not None and phy.get("packet_start_sample") is not None else None)
        older = (position is not None and self._latest_source_position is not None
                 and position[0] == self._latest_source_position[0]
                 and position[1] < self._latest_source_position[1])
        if older:
            self.out_of_order_results += 1
        else:
            self.latest = decoded
            self.envelope = envelope
            self._latest_source_position = position
            self.latest_phy_quality = phy.get("fragment_quality")
        counter = int(decoded["packet_counter"]) & 0xffff
        if not older:
            self.last_packet_counter = counter
        if self._counter_high_water is None:
            self._counter_high_water = counter
        else:
            delta = (counter-self._counter_high_water) & 0xffff
            if delta == 0:
                self.duplicate_counters += 1
            elif delta < 0x8000:
                self.estimated_counter_gaps += delta-1
                # Bound late-fill bookkeeping; the cumulative estimate remains.
                for step in range(max(1, delta-1024), delta):
                    self._pending_gaps.add((self._counter_high_water+step) & 0xffff)
                self._pending_gaps = {v for v in self._pending_gaps if ((counter-v) & 0xffff) <= 1024}
                self._counter_high_water = counter
            elif counter in self._pending_gaps:
                self._pending_gaps.remove(counter)
                self.estimated_counter_gaps -= 1
            else:
                # Backward/restarted counters are ambiguous. Keep the high-water
                # reference; never infer gaps from a backward transition. A real
                # reboot can therefore undercount gaps until it catches up.
                if not older:
                    self.counter_discontinuities += 1
        source_time = envelope.get("source_time_s")
        basis = str(envelope.get("source_id", "IQ")) if source_time is not None else "received_epoch"
        arrival = float(source_time) if source_time is not None else timestamp
        if self.recent_time_basis != basis:
            self.recent_arrivals.clear()
        self.recent_time_basis = basis
        self.recent_arrivals.append(arrival)
        if source_time is not None:
            self.recent_arrivals = deque(sorted(self.recent_arrivals), maxlen=1024)

    def recent_packet_rate_hz(self, window_s: float = 60., now: float | None = None) -> float:
        """Received packets / fixed window, in IQ time for replay, wall time live.

        This measures received packet rate; it makes no transmission-rate or
        delivery-percentage claim. ``now`` must use recent_time_basis.
        """
        if window_s <= 0 or not self.recent_arrivals:
            return 0.
        if now is None:
            now = time.time() if self.recent_time_basis == "received_epoch" else self.recent_arrivals[-1]
        return sum(0 <= now-t <= window_s for t in self.recent_arrivals)/window_s


@dataclass(slots=True)
class PacketUpdate:
    """Small summary returned after a packet is committed to the session."""

    packet_type: str
    phy_ok: bool
    app_ok: bool | None
    csid: int | None = None
    fleet_updated: bool = False


class FleetSession:
    """In-memory state for one GUI run.

    The receiver emits packet envelopes.  ``FleetSession`` commits each envelope
    exactly once, updates counters, and stores only CRC-valid DeSCENT telemetry
    in per-ChipSat state.
    """

    def __init__(self) -> None:
        self.fleet: dict[int, ChipSatState] = {}
        self.histories: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_POINTS)
        )
        self.packet_records = deque(maxlen=MAX_PACKET_RECORDS)
        self.valid_arrivals = deque()
        self.total_phy_ok = 0
        self.total_phy_fail = 0
        self.app_crc_pass = 0
        self.app_crc_fail = 0
        self.receiver_stages: dict[str, int] = {}

    def update_receiver_stats(self, counters: dict[str, int]) -> None:
        """Latest source-run PHY snapshot; session application totals stay separate."""
        self.receiver_stages = dict(counters)

    def diagnostics(self) -> dict[str, Any]:
        return {"source_run": dict(self.receiver_stages),
                "session": {"phy_crc_pass": self.total_phy_ok, "phy_crc_fail": self.total_phy_fail,
                            "application_crc_pass": self.app_crc_pass,
                            "application_crc_fail": self.app_crc_fail}}

    def add_packet(self, envelope: dict[str, Any], *, ignore_csid_zero: bool) -> PacketUpdate:
        """Commit one receiver envelope and return the state change."""
        phy_ok = bool(envelope.get("phy", {}).get("crc_ok", False))
        if phy_ok:
            self.total_phy_ok += 1
        else:
            self.total_phy_fail += 1

        self.packet_records.append(envelope)
        if phy_ok:
            now = float(envelope.get("received_epoch", time.time()))
            self.valid_arrivals.append(now)
            while self.valid_arrivals and self.valid_arrivals[0] < now-60:
                self.valid_arrivals.popleft()

        record = envelope.get("record", {})
        packet_type = str(record.get("packet_type", "UNKNOWN"))
        if phy_ok and packet_type == "DeSCENT_V2_5":
            self.app_crc_pass += 1
            app_ok: bool | None = True
        elif phy_ok and packet_type == "DeSCENT_V2_5_CRC_FAIL":
            self.app_crc_fail += 1
            app_ok = False
        else:
            app_ok = None

        update = PacketUpdate(packet_type=packet_type, phy_ok=phy_ok, app_ok=app_ok)
        if not phy_ok or packet_type != "DeSCENT_V2_5" or "decoded" not in record:
            return update

        decoded = record["decoded"]
        csid = int(decoded["csid"])
        update.csid = csid
        if ignore_csid_zero and csid == 0:
            return update

        now = float(envelope.get("received_epoch", time.time()))
        state = self.fleet.get(csid)
        if state is None:
            state = ChipSatState(csid=csid, first_seen=now, last_seen=now)
            self.fleet[csid] = state
        state.update(decoded, envelope, now)
        self.histories[csid].append({"t": now, "d": decoded})
        self.prune_history(csid, now)
        update.fleet_updated = True
        return update

    def prune_history(self, csid: int, now: float) -> None:
        history = self.histories.get(csid)
        while history and history[0]["t"] < now-300.0:
            history.popleft()

    def clear(self) -> None:
        self.fleet.clear()
        self.histories.clear()
        self.packet_records.clear()
        self.valid_arrivals.clear()
        self.total_phy_ok = 0
        self.total_phy_fail = 0
        self.app_crc_pass = 0
        self.app_crc_fail = 0
        self.receiver_stages.clear()

    def active_count(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return sum(1 for state in self.fleet.values() if now - state.last_seen < STALE_SECONDS)

    def crc_valid_rate_hz(self, *, window_s: float = 60.0, now: float | None = None) -> float:
        now = time.time() if now is None else now
        if window_s <= 0:
            return 0.0
        if window_s <= 60:
            return sum(now-window_s <= t <= now for t in self.valid_arrivals) / window_s
        recent = sum(
            1
            for envelope in self.packet_records
            if envelope.get("phy", {}).get("crc_ok")
            and now - float(envelope.get("received_epoch", 0.0)) <= window_s
        )
        return recent / window_s

    def csv_rows(self) -> list[dict[str, Any]]:
        """Flatten CRC-valid DeSCENT records for CSV export."""
        return list(self.csv_stream(self.packet_records))

    @staticmethod
    def csv_stream(envelopes):
        for envelope in envelopes:
            record = envelope.get("record", {})
            if (not envelope.get("phy", {}).get("crc_ok") or
                    record.get("packet_type") != "DeSCENT_V2_5" or "decoded" not in record):
                continue
            decoded = record["decoded"]
            phy = envelope.get("phy", {})
            ax, ay, az = decoded["linear_acceleration_mps2"]
            gx, gy, gz = decoded["gyroscope_deg_s"]
            mx, my, mz = decoded["magnetic_field_uT"]
            yield {
                "received_at": envelope.get("received_at"),
                "csid": decoded["csid"],
                "packet_counter": decoded["packet_counter"],
                "soc_percent": decoded["state_of_charge_percent"],
                "latitude_deg": decoded["latitude_deg"],
                "longitude_deg": decoded["longitude_deg"],
                "gps_altitude_msl_m": decoded["gps_altitude_msl_m"],
                "environmental_altitude_m": decoded["environmental_altitude_m"],
                "temperature_C": decoded["temperature_C"],
                "pressure_hPa": decoded["pressure_hPa"],
                "humidity_percent": decoded["humidity_percent"],
                "ax_mps2": ax,
                "ay_mps2": ay,
                "az_mps2": az,
                "gx_deg_s": gx,
                "gy_deg_s": gy,
                "gz_deg_s": gz,
                "mx_uT": mx,
                "my_uT": my,
                "mz_uT": mz,
                "validity_raw": decoded["validity_raw"],
                "all_data_fresh": decoded["valid"]["all_data_fresh"],
                "cfo_hz": phy.get("cfo_hz"),
                "fragments": phy.get("fragments"),
                "raw_hex": record.get("raw_hex"),
            }
