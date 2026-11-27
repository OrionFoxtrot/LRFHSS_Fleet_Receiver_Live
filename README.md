# LR-FHSS Fleet Receiver Live

A Windows receiver and Fleet Manager for DeSCENT ChipSat LR-FHSS telemetry. It
receives live RTL-SDR IQ or replays recordings, checks packet CRCs, and displays
per-ChipSat sensor values, validity, health and reception history.

## Tested system

Windows 11, 64-bit Python **3.13.17**, RTL-SDR with the Osmocom
**rtl-sdr-64bit-20260906** runtime, and an Intel NUC i7-1165G7
(4 cores / 8 threads). In November 2026 the default receiver was checked against
the full test suite, capture known answers, shutdown checks and paced 1x replay
of a 100-ChipSat fleet transmitting in 200 ms slots on a 20 s cycle. Results
apply to that configuration and host; they are not a loss-free reception guarantee.
The full test suite retains one known frozen-source comparison failure.
The public test subset (`run_tests.bat`) runs without the full test-data bundle.

## Install and start

Install 64-bit Python 3.13 from python.org; its `py` launcher creates the
environment. Use a short checkout path
(for example `C:\LRFHSS`): deeply nested folders can exceed Windows path limits
while installing PySide6. In PowerShell, from the extracted release or clone,
create a local environment:

```powershell
py -3.13 -m venv .venv
Copy-Item .venv\Scripts\python.exe .venv\Scripts\python3.exe
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\install.bat
.\run_gui.bat
```

If your Python came from the Microsoft Store, which provides `python3` but not
`py`, use `python3 -m venv .venv` instead.
The local `python3.exe` copy is the Windows venv launcher; the PATH change lasts
only for this shell. Launch the BAT files from this shell to keep using the venv.
`requirements.txt` is the normal install; `requirements-tested.txt` records the
validated Windows versions for optional reproducibility.

The supplied `rtl_tcp_fleet.exe` needs `librtlsdr.dll`, `libusb-1.0.dll` and
`libwinpthread-1.dll`. These are included in Osmocom's 64-bit Windows rtl-sdr build
`rtl-sdr-64bit-20260906.zip` (the version tested), from
<https://ftp.osmocom.org/binaries/windows/rtl-sdr/>. Copy the three DLLs from that
zip next to `rtl_tcp_fleet.exe`. Install the WinUSB driver for the dongle (for
example with Zadig) first. The DLLs are not redistributed in our release ZIP;
see `WINDOWS_DLLS.json` and the third-party notices.
Alternatively set `$env:RTL_TCP_EXE` to an installed helper before opening the GUI.
The program does not install drivers or change Windows security settings.

## Default and experimental modes

Default: **Single hop ID 256, SIC on, 40 dB live gain**, 3 dB trigger, 915 MHz
centre, 1 MS/s, and carrier search +/-4.5 kHz. Saved gain settings take precedence.
Confirm frequency and gain for the actual transmitter and RF conditions.

**Experimental:** Two IDs / rr 2 (256/299 initially), other hop IDs, and
weak-signal search. Two IDs starts at a 10 dB trigger with SIC disabled. The
transmitter's hop IDs must match the receiver. These are not the default
configuration covered by the shared-ID qualification above.

## CLI and recordings

```powershell
.\run_cli.bat --gain 40 --freq 915000000
.\decode_iq.bat .\recording.iq
.\decode_iq.bat .\recording.iq --exhaustive-file
```

IQ files are unsigned 8-bit interleaved I/Q at 1,000,000 complex samples/s.
The GUI has File input and an **Exhaustive decode (no real-time deadlines)**
checkbox. Exhaustive mode waits for pending work and enables the existing retry
grid; it may take longer than the recording and cannot recover absent samples.
It is for files only. Normal file replay is the default.

## Output and CSV

GUI session journals, sidecars and settings are stored in `logs/`; optional raw
IQ recordings and recording sidecars are under `logs/iq/`. Journals preserve packet
bytes and CRC results. Export CSV flattens PHY-CRC-valid DeSCENT V2.5 packets.
The DeSCENT V2.5 payload format comes from the DeSCENT ChipSat project:
<https://github.com/OrionFoxtrot/SSDS_DeSCENT>.
The CLI prints decoded packets; add `--record-iq PATH` to live capture to save IQ.

CSV columns (in order): `received_at`, `csid`, `packet_counter`, `soc_percent`,
`latitude_deg`, `longitude_deg`, `gps_altitude_msl_m`, `environmental_altitude_m`,
`temperature_C`, `pressure_hPa`, `humidity_percent`, `ax_mps2`, `ay_mps2`,
`az_mps2`, `gx_deg_s`, `gy_deg_s`, `gz_deg_s`, `mx_uT`, `my_uT`, `mz_uT`,
`validity_raw`, `all_data_fresh`, `cfo_hz`, `fragments`, `raw_hex`.
Sensor fields can be invalid or stale: use the validity fields, not just numeric
values. Logs and recordings may contain locations and identifiers; review them
before sharing.

## Public tests and sample asset

Run `run_tests.bat` for the public subset, without the full test-data bundle.
A capture test skips with instructions when the optional
`30s_lrfhss_test.iq` release asset is absent. Once an asset is published, download
it from the matching release into `tests/fixtures/` and run the tests again.
It is **60,000,000 bytes** (about 57.22 MiB), contains two default-decoded packets,
and is deliberately excluded from Git and the software ZIP.

SHA-256: `f09852865ad54b1e110aad6a104a71972158c686c88e39d4ddadc50ab7bdb3a8`.
It is attached to each GitHub release as a separate download.

## Limits and licensing

Reception depends on RF level, collisions, clock/frequency error, CPU headroom and
input continuity. Raw `rtl_tcp` has no true source-drop counter. GUI PRR is an
estimate; ADC readings are unavailable for file input. IQ recording needs about
7.2 GB/hour and can drop recording data if storage cannot keep up. A received CRC
is an integrity check, not authentication. This receiver does not support every
LR-FHSS variant or every application payload.

Python receiver code: [MIT License](LICENSE). The `rtl_tcp_fleet.exe` helper is
GPL-2.0-or-later; its upstream source, license, build script and local patch are
included under `native/rtl_tcp/`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the Semtech Clear BSD notice and DLL origin/licence references. Python
dependencies retain their own licenses and are installed separately.
