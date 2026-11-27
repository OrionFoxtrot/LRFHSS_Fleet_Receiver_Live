from __future__ import annotations

import json
import struct
from typing import Any, Mapping

PACKET_LENGTH = 55
DATA_LENGTH = 53

# DeSCENT V2.5 binary telemetry layout (little endian).
PACKET_STRUCT = struct.Struct("<4i3h3h3h4hhHHHBBBH")


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16-CCITT-FALSE: poly 0x1021, init 0xFFFF, xorout 0."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def decode_descent_packet(payload: bytes) -> dict[str, Any]:
    """Decode one CRC-valid 55-byte DeSCENT V2.5 telemetry packet."""
    if len(payload) != PACKET_LENGTH:
        raise ValueError(
            f"Expected {PACKET_LENGTH} bytes, received {len(payload)}"
        )

    received_crc = int.from_bytes(payload[53:55], "little")
    calculated_crc = crc16_ccitt_false(payload[:53])
    if received_crc != calculated_crc:
        raise ValueError(
            f"Application CRC mismatch: received 0x{received_crc:04X}, "
            f"calculated 0x{calculated_crc:04X}"
        )

    (
        latitude_e7,
        longitude_e7,
        gps_altitude_msl_mm,
        environmental_altitude_cm,
        ax_raw, ay_raw, az_raw,
        gx_raw, gy_raw, gz_raw,
        mx_raw, my_raw, mz_raw,
        qi_raw, qj_raw, qk_raw, qr_raw,
        temperature_raw,
        pressure_raw,
        humidity_raw,
        packet_counter,
        csid,
        soc_x2,
        validity,
        crc_field,
    ) = PACKET_STRUCT.unpack(payload)

    valid = {
        "linear_acceleration": bool(validity & 0x01),
        "gyroscope": bool(validity & 0x02),
        "magnetometer": bool(validity & 0x04),
        "quaternion": bool(validity & 0x08),
        "gps": bool(validity & 0x10),
        "state_of_charge": bool(validity & 0x20),
        "environmental": bool(validity & 0x40),
        "all_data_fresh": bool(validity & 0x80),
    }

    return {
        "packet_type": "DeSCENT_V2_5",
        "length_bytes": len(payload),
        "application_crc_ok": True,
        "application_crc16": crc_field,
        "csid": csid,
        "packet_counter": packet_counter,
        "validity_raw": validity,
        "valid": valid,
        "latitude_deg": latitude_e7 / 1e7,
        "longitude_deg": longitude_e7 / 1e7,
        "gps_altitude_msl_m": gps_altitude_msl_mm / 1000.0,
        "environmental_altitude_m": environmental_altitude_cm / 100.0,
        "linear_acceleration_mps2": (
            ax_raw / 100.0,
            ay_raw / 100.0,
            az_raw / 100.0,
        ),
        "gyroscope_deg_s": (
            gx_raw / 10.0,
            gy_raw / 10.0,
            gz_raw / 10.0,
        ),
        "magnetic_field_uT": (
            mx_raw / 10.0,
            my_raw / 10.0,
            mz_raw / 10.0,
        ),
        "quaternion_ijkr": (
            qi_raw / 32767.0,
            qj_raw / 32767.0,
            qk_raw / 32767.0,
            qr_raw / 32767.0,
        ),
        "temperature_C": temperature_raw / 100.0,
        "pressure_hPa": pressure_raw / 10.0,
        "humidity_percent": humidity_raw / 100.0,
        "state_of_charge_percent": soc_x2 / 2.0,
    }


def raw_hex(payload: bytes) -> str:
    return " ".join(f"{b:02X}" for b in payload)


def safe_ascii(payload: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in payload)


def _mostly_printable(text: str) -> bool:
    if not text:
        return False
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    return printable / len(text) >= 0.90


def _parse_key_value_ascii(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in text.replace("\r", "").replace("\n", ",").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def payload_to_record(
    payload: bytes,
    *,
    rf_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one JSON-serializable decoded packet record."""
    record: dict[str, Any] = {
        "raw_hex": raw_hex(payload),
        "raw_length_bytes": len(payload),
    }
    if rf_meta:
        record["rf"] = dict(rf_meta)

    # Printable telemetry gets priority, even if it happens to be exactly 55 B.
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = ""

    if text and _mostly_printable(text):
        record["packet_type"] = "ASCII"
        record["ascii"] = text
        fields = _parse_key_value_ascii(text)
        if fields:
            record["fields"] = fields
        return record

    # Otherwise, a 55-byte payload is treated as the DeSCENT binary schema.
    if len(payload) == PACKET_LENGTH:
        try:
            record["decoded"] = decode_descent_packet(payload)
            return record
        except ValueError as exc:
            record["packet_type"] = "DeSCENT_V2_5_CRC_FAIL"
            record["decode_error"] = str(exc)
            return record

    record["packet_type"] = "BINARY"
    record["ascii_preview"] = safe_ascii(payload)
    return record


def _valid_label(valid: bool) -> str:
    return "VALID" if valid else "INVALID/STALE"


def format_payload(
    payload: bytes,
    *,
    rf_meta: Mapping[str, Any] | None = None,
) -> str:
    """Human-readable output. The complete raw packet is always included."""
    record = payload_to_record(payload, rf_meta=rf_meta)
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("LR-FHSS PACKET")
    lines.append("=" * 72)

    if rf_meta:
        if rf_meta.get("timestamp") is not None:
            lines.append(f"RX time:       {rf_meta['timestamp']}")
        if rf_meta.get("rssi_dbfs") is not None:
            lines.append(f"RSSI:          {rf_meta['rssi_dbfs']:.1f} dBFS")
        if rf_meta.get("snr_db") is not None:
            lines.append(f"SNR:           {rf_meta['snr_db']:.1f} dB")
        if rf_meta.get("cfo_hz") is not None:
            lines.append(f"CFO:           {rf_meta['cfo_hz']:+.0f} Hz")

    lines.append(f"Length:        {len(payload)} bytes")
    lines.append("")
    lines.append("RAW PACKET")
    lines.append(f"  {raw_hex(payload)}")

    packet_type = record.get("packet_type")

    if packet_type == "ASCII":
        lines.append("")
        lines.append("DECODED DATA — ASCII")
        fields = record.get("fields", {})
        if fields:
            for key, value in fields.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {record['ascii']}")
        return "\n".join(lines)

    if packet_type == "DeSCENT_V2_5_CRC_FAIL":
        lines.append("")
        lines.append("DECODED DATA")
        lines.append("  DeSCENT V2.5 packet detected, but application CRC FAILED.")
        lines.append(f"  {record['decode_error']}")
        lines.append("  Engineering-unit values are intentionally not published.")
        return "\n".join(lines)

    if "decoded" in record:
        d = record["decoded"]
        v = d["valid"]

        lines.append("")
        lines.append("DECODED DATA — DeSCENT V2.5")
        lines.append(f"  Application CRC: PASS (0x{d['application_crc16']:04X})")
        lines.append(f"  CSID:            {d['csid']}")
        lines.append(f"  Packet counter:  {d['packet_counter']}")
        lines.append(
            f"  Freshness:       {'ALL DATA FRESH' if v['all_data_fresh'] else 'NOT ALL DATA FRESH'}"
        )
        lines.append(f"  Validity byte:   0x{d['validity_raw']:02X}")
        lines.append("")

        lines.append(f"  GNSS [{_valid_label(v['gps'])}]")
        lines.append(f"    Latitude:      {d['latitude_deg']:.7f} deg")
        lines.append(f"    Longitude:     {d['longitude_deg']:.7f} deg")
        lines.append(f"    Altitude MSL:  {d['gps_altitude_msl_m']:.3f} m")

        lines.append("")
        lines.append(f"  Environment [{_valid_label(v['environmental'])}]")
        lines.append(f"    Temperature:   {d['temperature_C']:.2f} C")
        lines.append(f"    Pressure:      {d['pressure_hPa']:.1f} hPa")
        lines.append(f"    Humidity:      {d['humidity_percent']:.2f} %")
        lines.append(f"    Pressure alt:  {d['environmental_altitude_m']:.2f} m")

        lines.append("")
        lines.append(f"  Power [{_valid_label(v['state_of_charge'])}]")
        lines.append(f"    State of charge: {d['state_of_charge_percent']:.1f} %")

        ax, ay, az = d["linear_acceleration_mps2"]
        lines.append("")
        lines.append(f"  Linear acceleration [{_valid_label(v['linear_acceleration'])}]")
        lines.append(f"    X/Y/Z:         {ax:.2f}, {ay:.2f}, {az:.2f} m/s^2")

        gx, gy, gz = d["gyroscope_deg_s"]
        lines.append("")
        lines.append(f"  Gyroscope [{_valid_label(v['gyroscope'])}]")
        lines.append(f"    X/Y/Z:         {gx:.1f}, {gy:.1f}, {gz:.1f} deg/s")

        mx, my, mz = d["magnetic_field_uT"]
        lines.append("")
        lines.append(f"  Magnetometer [{_valid_label(v['magnetometer'])}]")
        lines.append(f"    X/Y/Z:         {mx:.1f}, {my:.1f}, {mz:.1f} uT")

        qi, qj, qk, qr = d["quaternion_ijkr"]
        lines.append("")
        lines.append(f"  Quaternion [{_valid_label(v['quaternion'])}]")
        lines.append(f"    I/J/K/Real:    {qi:.5f}, {qj:.5f}, {qk:.5f}, {qr:.5f}")

        return "\n".join(lines)

    lines.append("")
    lines.append("DECODED DATA")
    lines.append("  Generic binary payload; no known packet schema matched.")
    lines.append(f"  ASCII preview: {record.get('ascii_preview', '')}")
    return "\n".join(lines)


def record_to_json(
    payload: bytes,
    *,
    rf_meta: Mapping[str, Any] | None = None,
    indent: int | None = 2,
) -> str:
    """JSON output for future serial, UDP, CSV, or GUI use."""
    return json.dumps(payload_to_record(payload, rf_meta=rf_meta), indent=indent)


def print_payload(
    payload: bytes,
    *,
    rf_meta: Mapping[str, Any] | None = None,
) -> None:
    print(format_payload(payload, rf_meta=rf_meta))
