from __future__ import annotations

import json
import struct
from typing import Any, Mapping

PACKET_LENGTH = 55
PACKET_STRUCT = struct.Struct("<4i3h3h3h4hhHHHBBBH")


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def decode_descent_packet(payload: bytes) -> dict[str, Any]:
    if len(payload) != PACKET_LENGTH:
        raise ValueError(f"Expected 55 bytes, received {len(payload)}")

    rx_crc = int.from_bytes(payload[53:55], "little")
    calc_crc = crc16_ccitt_false(payload[:53])
    if rx_crc != calc_crc:
        raise ValueError(
            f"Application CRC mismatch: received 0x{rx_crc:04X}, calculated 0x{calc_crc:04X}"
        )

    (
        latitude_e7, longitude_e7, gps_altitude_msl_mm, environmental_altitude_cm,
        ax_raw, ay_raw, az_raw,
        gx_raw, gy_raw, gz_raw,
        mx_raw, my_raw, mz_raw,
        qi_raw, qj_raw, qk_raw, qr_raw,
        temperature_raw, pressure_raw, humidity_raw,
        packet_counter, csid, soc_x2, validity, crc_field,
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
        "linear_acceleration_mps2": (ax_raw / 100.0, ay_raw / 100.0, az_raw / 100.0),
        "gyroscope_deg_s": (gx_raw / 10.0, gy_raw / 10.0, gz_raw / 10.0),
        "magnetic_field_uT": (mx_raw / 10.0, my_raw / 10.0, mz_raw / 10.0),
        "quaternion_ijkr": (
            qi_raw / 32767.0, qj_raw / 32767.0, qk_raw / 32767.0, qr_raw / 32767.0
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


def _ascii_text(payload: bytes) -> str | None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    return text if printable / len(text) >= 0.90 else None


def _parse_ascii_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for piece in text.replace("\r", "").replace("\n", ",").split(","):
        piece = piece.strip()
        if ":" in piece:
            key, value = piece.split(":", 1)
            if key.strip():
                fields[key.strip()] = value.strip()
    return fields


def payload_to_record(payload: bytes, *, rf_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "raw_length_bytes": len(payload),
        "raw_hex": raw_hex(payload),
    }
    if rf_meta:
        record["rf"] = dict(rf_meta)

    # Check for readable ASCII first, including 55-byte ASCII test strings.
    text = _ascii_text(payload)
    if text is not None:
        record["packet_type"] = "ASCII"
        record["ascii"] = text
        fields = _parse_ascii_fields(text)
        if fields:
            record["fields"] = fields
        return record

    # Then try the real 55-byte DeSCENT binary telemetry format.
    if len(payload) == PACKET_LENGTH:
        try:
            record["packet_type"] = "DeSCENT_V2_5"
            record["decoded"] = decode_descent_packet(payload)
        except ValueError as exc:
            record["packet_type"] = "DeSCENT_V2_5_CRC_FAIL"
            record["decode_error"] = str(exc)
        return record

    record["packet_type"] = "BINARY"
    record["ascii_preview"] = safe_ascii(payload)
    return record


def _state(ok: bool) -> str:
    return "VALID" if ok else "INVALID/STALE"


def format_payload(payload: bytes, *, rf_meta: Mapping[str, Any] | None = None) -> str:
    r = payload_to_record(payload, rf_meta=rf_meta)
    lines = [
        "=" * 72,
        "LR-FHSS PACKET",
        "=" * 72,
    ]

    if rf_meta:
        if rf_meta.get("timestamp") is not None:
            lines.append(f"RX time:       {rf_meta['timestamp']}")
        if rf_meta.get("rssi_dbfs") is not None:
            lines.append(f"RSSI:          {rf_meta['rssi_dbfs']:.1f} dBFS")
        if rf_meta.get("snr_db") is not None:
            lines.append(f"SNR:           {rf_meta['snr_db']:.1f} dB")
        if rf_meta.get("cfo_hz") is not None:
            lines.append(f"CFO:           {rf_meta['cfo_hz']:+.0f} Hz")

    lines += [
        f"Length:        {len(payload)} bytes",
        "",
        "RAW PACKET",
        f"  {raw_hex(payload)}",
    ]

    if r["packet_type"] == "ASCII":
        lines += ["", "DECODED DATA — ASCII"]
        if r.get("fields"):
            for k, v in r["fields"].items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {r['ascii']}")
        return "\n".join(lines)

    if r["packet_type"] == "DeSCENT_V2_5_CRC_FAIL":
        lines += [
            "",
            "DECODED DATA",
            "  DeSCENT V2.5-sized packet, but application CRC FAILED.",
            f"  {r['decode_error']}",
            "  Raw bytes are preserved; engineering values are not published.",
        ]
        return "\n".join(lines)

    if r["packet_type"] == "DeSCENT_V2_5":
        d = r["decoded"]
        v = d["valid"]
        ax, ay, az = d["linear_acceleration_mps2"]
        gx, gy, gz = d["gyroscope_deg_s"]
        mx, my, mz = d["magnetic_field_uT"]
        qi, qj, qk, qr = d["quaternion_ijkr"]

        lines += [
            "",
            "DECODED DATA — DeSCENT V2.5",
            f"  Application CRC: PASS (0x{d['application_crc16']:04X})",
            f"  CSID:            {d['csid']}",
            f"  Packet counter:  {d['packet_counter']}",
            f"  Freshness:       {'ALL DATA FRESH' if v['all_data_fresh'] else 'NOT ALL DATA FRESH'}",
            f"  Validity byte:   0x{d['validity_raw']:02X}",
            "",
            f"  GNSS [{_state(v['gps'])}]",
            f"    Latitude:      {d['latitude_deg']:.7f} deg",
            f"    Longitude:     {d['longitude_deg']:.7f} deg",
            f"    Altitude MSL:  {d['gps_altitude_msl_m']:.3f} m",
            "",
            f"  Environment [{_state(v['environmental'])}]",
            f"    Temperature:   {d['temperature_C']:.2f} C",
            f"    Pressure:      {d['pressure_hPa']:.1f} hPa",
            f"    Humidity:      {d['humidity_percent']:.2f} %",
            f"    Pressure alt:  {d['environmental_altitude_m']:.2f} m",
            "",
            f"  Power [{_state(v['state_of_charge'])}]",
            f"    State of charge: {d['state_of_charge_percent']:.1f} %",
            "",
            f"  Linear acceleration [{_state(v['linear_acceleration'])}]",
            f"    X/Y/Z:         {ax:.2f}, {ay:.2f}, {az:.2f} m/s^2",
            "",
            f"  Gyroscope [{_state(v['gyroscope'])}]",
            f"    X/Y/Z:         {gx:.1f}, {gy:.1f}, {gz:.1f} deg/s",
            "",
            f"  Magnetometer [{_state(v['magnetometer'])}]",
            f"    X/Y/Z:         {mx:.1f}, {my:.1f}, {mz:.1f} uT",
            "",
            f"  Quaternion [{_state(v['quaternion'])}]",
            f"    I/J/K/Real:    {qi:.5f}, {qj:.5f}, {qk:.5f}, {qr:.5f}",
        ]
        return "\n".join(lines)

    lines += [
        "",
        "DECODED DATA",
        "  Generic binary payload; no known schema matched.",
        f"  ASCII preview: {r.get('ascii_preview', '')}",
    ]
    return "\n".join(lines)


def record_to_json(
    payload: bytes,
    *,
    rf_meta: Mapping[str, Any] | None = None,
    indent: int | None = 2,
) -> str:
    return json.dumps(payload_to_record(payload, rf_meta=rf_meta), indent=indent)


def print_payload(payload: bytes, *, rf_meta: Mapping[str, Any] | None = None) -> None:
    print(format_payload(payload, rf_meta=rf_meta))
