#!/usr/bin/env python3
"""Receiver-thread adapter used by the Fleet Manager.

This module owns source I/O and converts decoded PHY packets into GUI-neutral
packet envelopes. It does not own fleet state or widgets.
"""
from __future__ import annotations

import os
import shutil
import socket
import select
import struct
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from lrfhss_live import FS, LiveDecoder, Packet, TruncatedIQError, hop_frequencies
from lrfhss_readable import payload_to_record
from lrfhss_source_monitor import SourceMonitor
from lrfhss_iq_recorder import IQRecorder, default_recording_path

from .config import (
    FLEET_RTL_TCP, LIVE_USB_BYTES, SHORT_USB_BYTES, ReceiverConfig,
    decoder_options, rtl_tcp_fallbacks, HOP_RR2,
)

RECORDING_DIR = Path(__file__).resolve().parents[2] / "logs" / "iq"


class RtlTcpPolicyError(RuntimeError):
    def __init__(self, executable: str):
        super().__init__(
            f'Windows Application Control blocked "{executable}" or one of its runtime DLLs. '
            "Set RTL_TCP_EXE to a trusted RTL-SDR installation's rtl_tcp.exe. "
            "The Microsoft-Windows-CodeIntegrity/Operational event log identifies the blocked file."
        )


class ReceiverWorker(QtCore.QObject):
    packet = QtCore.Signal(object)
    receiver_stats = QtCore.Signal(object)
    message = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()
    autoset_result = QtCore.Signal(object)

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config
        self.publish_packet = self.packet.emit
        self.stop_event = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.total_phy_ok = 0
        self.total_phy_fail = 0
        self.bytes_read = 0
        self.started_wall = time.time()
        self.last_stats = 0.0
        self.last_detail = 0.0
        self.file_progress = None
        self.decoder: LiveDecoder | None = None
        self.source_id = f"{config.mode}:{config.file_path}:{self.started_wall}"
        # rtl_tcp diagnostics: drained continuously (a full pipe would block the SDR
        # process and stall IQ delivery); the last lines are kept for error messages.
        self.stderr_tail: deque[str] = deque(maxlen=50)
        self.stderr_thread: threading.Thread | None = None
        # Live-source health (no-data watchdog, ADC level); live mode only.
        self.source_monitor: SourceMonitor | None = None
        self.recorder: IQRecorder | None = None
        self._recording_warned = False

    def request_stop(self) -> None:
        self.stop_event.set()
        # Closing the socket wakes a blocking recv immediately.
        try:
            if self.sock is not None:
                self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self.config.mode == "live":
                self._run_live()
            else:
                self._run_file()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error.emit(f"Receiver stopped: {exc}")
        finally:
            try:
                self._cleanup()
            except Exception as exc:                      # never skip the finished signal
                self.error.emit(f"Receiver cleanup failed: {exc}")
            finally:
                self.finished.emit()

    def _emit_stats(self, *, status: str, force: bool = False) -> None:
        now = time.time()
        if now - self.last_stats < 0.20 and status == "RUNNING" and not force:
            return
        self.last_stats = now
        detail = {}
        if self.decoder is not None and (force or status != "RUNNING" or now-self.last_detail >= 1.0):
            if force or status != "RUNNING":
                detail["receiver_stages"] = self.decoder.receiver_stage_stats()
            else:
                detail["receiver_stages"] = {
                    "processed_samples": self.decoder.abs_sample,
                    "phy_crc_pass": self.total_phy_ok, "phy_crc_fail": self.total_phy_fail,
                }
                if self.decoder.payload_retry == "adaptive":
                    detail["receiver_stages"]["adaptive_retry"] = self.decoder.payload_pool.retry_stats()
            if self.config.hop_id_mode == HOP_RR2:
                monitor = getattr(self, "rr2_health", None)
                health = (monitor.snapshot(self.decoder.input_lag_s()) if monitor is not None else
                          dict(lag_s=0., lag_warning=False, estimated_source_drop_samples=None))
                health["job_rejections"] = sum(self.decoder.runtime_counts.get(key, 0)
                    for key in ("header_jobs_rejected", "payload_jobs_rejected"))
                detail["rr2_health"] = health
            if self.config.mode == "live":
                # Observe the same clock as the benchmark without calling
                # input_lag_s(): that method may adjust the decoder's origin.
                decoder = self.decoder
                clock = time.perf_counter()
                note = decoder._input_lag_note
                origin = decoder._input_clock_offset
                lag = (note[0] if note is not None and clock-note[1] <= 1.0 else
                       max(0., clock-decoder.input_abs_sample/FS-origin) if origin is not None else None)
                detail["decoder_health"] = {"lag_s": lag}
                if status == "RUNNING" and not force:
                    detail["receiver_stages"]["header_candidates_expired"] = decoder.runtime_counts.get("header_candidates_expired", 0)
            self.last_detail = now
        if self.file_progress is not None:
            detail["file_progress"] = self.file_progress
        exhaustive = getattr(self, "exhaustive", None)
        if exhaustive is not None:
            detail["exhaustive_file"] = exhaustive.snapshot(self.file_progress)
        self.receiver_stats.emit({
            "status": status, "phy_ok": self.total_phy_ok, "phy_fail": self.total_phy_fail,
            "bytes_read": self.bytes_read, "uptime_s": max(0.0, now-self.started_wall),
            "rtl_tcp": self.config.rtl_tcp if self.config.mode == "live" else None,
            "source": self.source_monitor.snapshot() if self.source_monitor is not None else {},
            "recording": self.recorder.stats() if self.recorder is not None else {}, **detail,
        })

    def _packet_record(self, pkt: Packet, center_hz: int, *, source_time: float | None = None) -> dict[str, Any]:
        wall = datetime.now().astimezone()
        try:
            hops = hop_frequencies(
                pkt.header.info["grid"],
                pkt.header.info["bw_code"],
                pkt.header.info["hop_seq_id"],
                3,
                pkt.fragment_count,
            )
            cfo_hz = float(pkt.header.freq_hz - hops[0])
        except Exception:
            cfo_hz = None

        detail = getattr(pkt, "payload_diagnostics", {})
        phy = {
            "crc_ok": bool(pkt.crc_ok),
            "fragments": int(pkt.fragment_count),
            "header_cost": float(pkt.header.cost),
            "viterbi_cost": float(pkt.viterbi_cost),
            "header_frequency_offset_hz": float(pkt.header.freq_hz),
            "cfo_hz": cfo_hz,
            "hop_seq_id": int(pkt.header.info.get("hop_seq_id", 0)),
            "payload_len": len(pkt.payload),
            "fragment_quality": detail.get("fragment_quality"),
            "payload_method": detail.get("selected_method"),
            "job_id": detail.get("job_id"),
            "receiver_id": detail.get("receiver_id"),
            "track_id": detail.get("track_id"),
            "packet_start_sample": pkt.header.start,
            "payload_decode_wall_s": detail.get("total_wall_s"),
        }
        retry = detail.get("payload_retry")
        if retry and retry.get("mode") == "adaptive":
            phy["payload_retry"] = retry
        envelope: dict[str, Any] = {
            "received_at": wall.isoformat(timespec="milliseconds"),
            "received_epoch": time.time(),
            "source_time_s": source_time,
            "source_id": self.source_id,
            "center_frequency_hz": int(center_hz),
            "phy": phy,
        }

        if pkt.crc_ok:
            envelope["record"] = payload_to_record(
                pkt.payload,
                rf_meta={
                    "timestamp": wall.strftime("%H:%M:%S.%f")[:-3],
                    "cfo_hz": cfo_hz,
                },
            )
        else:
            envelope["record"] = {
                "packet_type": "PHY_CRC_FAIL",
                "raw_length_bytes": len(pkt.payload),
                "raw_hex": " ".join(f"{b:02X}" for b in pkt.payload),
            }
        return envelope

    def _handle_packets(self, packets: list[Packet], center_hz: int, *, file_mode: bool = False) -> None:
        for pkt in packets:
            if pkt.crc_ok:
                self.total_phy_ok += 1
            else:
                self.total_phy_fail += 1
            source_time = pkt.header.start / FS if file_mode else None
            envelope = self._packet_record(pkt, center_hz, source_time=source_time)
            self.publish_packet(envelope)
            if self.decoder is not None:
                parsed = envelope["record"].get("decoded", {})
                self.decoder._acquisition_event("application_result_published", self.decoder.abs_sample,
                    job_id=pkt.payload_diagnostics.get("job_id"),
                    track_id=pkt.payload_diagnostics.get("track_id"),
                    packet_type=envelope["record"].get("packet_type"),
                    csid=parsed.get("csid"), packet_counter=parsed.get("packet_counter"))
        self._emit_stats(status="RUNNING")

    def _run_file(self) -> None:
        path = self.config.file_path
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"IQ file not found: {path}")
        self.message.emit(f"Decoding IQ file: {path}")
        self._emit_stats(status="RUNNING")
        dec = self.decoder = LiveDecoder(self.config.trigger_db, verbose=False, **decoder_options(self.config))
        block_bytes = 100_000
        if self.config.exhaustive_file:
            from lrfhss_exhaustive_file import ExhaustiveFile
            self.exhaustive = ExhaustiveFile(dec, warn=lambda text: self.message.emit("WARNING: " + text))
        size = max(1, os.path.getsize(path))
        with open(path, "rb") as f:
            while not self.stop_event.is_set():
                raw = f.read(block_bytes)
                if not raw:
                    break
                if self.config.exhaustive_file and not self.exhaustive.before_feed(
                    lambda: self._decode(dec.service_file_backlog, file_mode=True),
                    next_samples=len(raw)//2, stop_event=self.stop_event,
                    on_wait=lambda: self._emit_stats(status="RUNNING")):
                    break
                self.bytes_read += len(raw)
                self.file_progress = min(1.0, self.bytes_read / size)
                self._decode(lambda: dec.feed_raw(raw), file_mode=True)
        if not self.stop_event.is_set():
            reason = self._finish_stream(dec, file_mode=True)
            if reason not in ("finished", "receiver_stopped"):
                raise RuntimeError(f"IQ file decode incomplete: end-of-stream drain ended with {reason}")
            if reason == "finished":
                self.message.emit("IQ file decode complete.")
        self._emit_stats(status="STOPPED")

    def _record(self, raw: bytes) -> None:
        if self.recorder is None or not raw:
            return
        self.recorder.write(raw)
        if self.recorder.stopped_reason and not self._recording_warned:
            self._recording_warned = True
            self._warn([f"Raw IQ recording stopped ({self.recorder.stopped_reason}); decoding continues."])
        elif self.recorder.dropped and not self._recording_warned:
            self._recording_warned = True
            self._warn(["The disk is not keeping up: raw IQ blocks are being dropped from the recording "
                        "(gaps are listed in its .json file); decoding continues."])

    def _watch(self) -> None:
        """No-data watchdog: runs while the receiver waits for IQ."""
        if self.source_monitor is not None:
            self._warn(self.source_monitor.check())

    def _warn(self, messages) -> None:
        for text in messages:
            self.message.emit(f"WARNING: {text}")
        if messages:
            self._emit_stats(status="RUNNING", force=True)

    def _decode(self, call, *, file_mode: bool = False) -> None:
        """Run one packet-returning decoder call (feed_raw, service_idle) and show its
        packets. If it raises, the packets it had already collected are fetched with
        take_unreturned_packets() and shown before the error propagates (Audit 02 B5)."""
        try:
            packets = call()
        except BaseException:
            try:
                recovered = self.decoder.take_unreturned_packets() if self.decoder is not None else []
            except Exception:
                recovered = []
            if recovered:
                self._handle_packets(recovered, self.config.freq, file_mode=file_mode)
            raise
        self._handle_packets(packets, self.config.freq, file_mode=file_mode)

    def _finish_stream(self, dec: LiveDecoder, *, file_mode: bool) -> str:
        """Natural end of a source (file EOF, SDR stream end): bounded finish(), every
        packet delivered, the drain outcome returned (Batch 10.4k, review finding 3).
        Stop stays an immediate abort: finish() honours stop_event."""
        try:
            exhaustive = getattr(self, "exhaustive", None) if file_mode else None
            final = (exhaustive.finish(self.stop_event) if exhaustive is not None
                     else dec.finish(stop_event=self.stop_event))
        except TruncatedIQError as exc:
            # Change 11/12: a truncated trailing byte pair is reported, but
            # the packets decoded from every complete sample are still shown.
            self.message.emit(f"Warning: {exc}")
            final = dec.take_unreturned_packets()
        except Exception:
            # Any other failure keeps its error path; the packets already
            # decoded are still shown first.
            self._handle_packets(dec.take_unreturned_packets(), self.config.freq, file_mode=file_mode)
            raise
        self._handle_packets(final, self.config.freq, file_mode=file_mode)
        return dec.finish_reason or "finished"

    def _connect_rtl_tcp(self, exe: str) -> int | None:
        """Start one helper and connect, distinguishing Windows policy failures."""
        cmd = [
            exe,
            "-a", "127.0.0.1",
            "-p", str(self.config.port),
            "-f", str(self.config.freq),
            "-s", str(FS),
            "-g", str(self.config.gain),
            "-P", str(self.config.ppm),
        ]
        transfer_bytes = None
        if Path(exe).resolve() == FLEET_RTL_TCP:
            transfer_bytes = SHORT_USB_BYTES if self.config.autoset_only else self.config.usb_transfer_bytes
            if transfer_bytes not in (SHORT_USB_BYTES, LIVE_USB_BYTES):
                raise ValueError(f"Unsupported USB transfer size: {transfer_bytes}")
        if transfer_bytes == LIVE_USB_BYTES:
            # -M selects 1 MiB USB reads. Scale rtl_tcp's outgoing backlog from
            # 500 to 125 blocks to keep roughly the same worst-case queued bytes.
            cmd.extend(["-M", "-n", "125"])
        transfer_info = {LIVE_USB_BYTES: "1 MiB", SHORT_USB_BYTES: "256 KiB"}.get(transfer_bytes, "helper-default")
        self.message.emit(f"Starting RTL-SDR receiver ({transfer_info} USB transfers)...")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                creationflags=creationflags,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in (4551, 1260, 577):
                raise RtlTcpPolicyError(exe) from exc
            raise
        self.stderr_thread = threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True,
                                              name="rtl_tcp-stderr")
        self.stderr_thread.start()

        deadline = time.time() + 7.0
        while time.time() < deadline and not self.stop_event.is_set():
            returncode = self.proc.poll()
            if returncode is not None:
                if self.stderr_thread is not None:
                    self.stderr_thread.join(timeout=1.0)      # the drain thread reads to EOF
                err = " | ".join(self.stderr_tail)
                # A blocked DLL can terminate the child after Popen succeeds.
                if returncode & 0xffffffff in (0xc0e90002, 0xc0000428):
                    raise RtlTcpPolicyError(exe)
                raise RuntimeError(f"rtl_tcp exited before connection (exit {returncode}). {err}".strip())
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.config.port), timeout=0.4)
                break
            except OSError:
                time.sleep(0.12)

        if self.stop_event.is_set():
            return transfer_bytes
        if self.sock is None:
            raise RuntimeError(f"Could not connect to rtl_tcp on port {self.config.port}.")
        return transfer_bytes

    def _run_live(self) -> None:
        exe = shutil.which(self.config.rtl_tcp)
        if not exe:
            package_dir = Path(__file__).resolve().parents[2] / "lrfhss_gui_app"
            candidates = (
                package_dir / self.config.rtl_tcp,
                package_dir.parent / self.config.rtl_tcp,
            )
            for local in candidates:
                if local.is_file():
                    exe = str(local)
                    break
        if not exe:
            raise FileNotFoundError(
                f"Could not find {self.config.rtl_tcp!r}. Keep rtl_tcp.exe beside this program or set RTL_TCP_EXE."
            )

        try:
            transfer_bytes = self._connect_rtl_tcp(exe)
        except RtlTcpPolicyError as exc:
            bundled = (FLEET_RTL_TCP, FLEET_RTL_TCP.with_name("rtl_tcp.exe"))
            if Path(exe).resolve() not in bundled or os.environ.get("RTL_TCP_EXE"):
                raise
            errors = [str(exc)]
            self._cleanup()
            for fallback in rtl_tcp_fallbacks():
                if self.stop_event.is_set():
                    return
                if Path(fallback).resolve() == Path(exe).resolve():
                    continue
                self.message.emit(
                    f'Windows blocked "{exe}". Trying "{fallback}" with helper-default USB transfers.'
                )
                self.stderr_tail.clear()
                try:
                    transfer_bytes = self._connect_rtl_tcp(fallback)
                except RtlTcpPolicyError as fallback_error:
                    errors.append(str(fallback_error))
                    self._cleanup()
                    continue
                exe = fallback
                break
            else:
                raise RuntimeError("\n\n".join(errors)) from exc

        self.config.rtl_tcp = exe
        if self.stop_event.is_set():
            return

        self.sock.settimeout(0.6)
        info = self._recv_exact_interruptible(12)
        if len(info) != 12 or info[:4] != b"RTL0":
            raise RuntimeError("Did not receive a valid rtl_tcp dongle header.")

        if self.config.autoset_only:
            self._run_autoset(info)
            return

        self.message.emit(
            f"Connected: {self.config.freq/1e6:.3f} MHz | 1.000 MS/s | gain {self.config.gain:g} dB"
        )
        self._emit_stats(status="RUNNING")

        dec = self.decoder = LiveDecoder(self.config.trigger_db, verbose=False, **decoder_options(self.config))
        self.source_monitor = SourceMonitor()
        if self.config.hop_id_mode == HOP_RR2:
            from .rr2_health import RR2Health
            self.rr2_health = RR2Health(FS, transfer_bytes)
        if self.config.record_iq:
            path = default_recording_path(RECORDING_DIR, self.config.freq, self.config.gain)
            self.recorder = IQRecorder(path, metadata=dict(center_hz=self.config.freq, gain_db=self.config.gain,
                                                           ppm=self.config.ppm, source="rtl_tcp", usb_transfer_bytes=transfer_bytes,
                                                           rtl_tcp=exe,
                                                           receiver="LR-FHSS Fleet Manager"))
            self.message.emit(f"Recording raw IQ to {path}")
        block_bytes = 100_000
        while not self.stop_event.is_set():
            try:
                raw = self._recv_exact_interruptible(block_bytes)
            except (ConnectionError, OSError) as exc:
                if self.stop_event.is_set():
                    break
                try:
                    partial = getattr(exc, "partial", b"")
                    self._record(partial)
                    if partial:
                        self.bytes_read += len(partial)
                        try:
                            self._handle_packets(dec.feed_raw(partial), self.config.freq)
                        except Exception:
                            self._handle_packets(dec.take_unreturned_packets(), self.config.freq)
                            raise
                    self._finish_stream(dec, file_mode=False)     # decode what was received, then report
                except Exception as drain_exc:
                    self.message.emit(f"End-of-stream drain failed: {drain_exc}")
                raise
            if not raw:
                if self.stop_event.is_set():
                    break
                # The SDR stream ended on its own: finish the received samples and
                # admitted work first (10.4k), then report the end as an error.
                self.message.emit("RTL-SDR stream ended; finishing received packets...")
                reason = self._finish_stream(dec, file_mode=False)
                if reason not in ("finished", "receiver_stopped"):
                    raise ConnectionError(f"RTL-SDR stream ended; end-of-stream drain incomplete ({reason}).")
                raise ConnectionError("RTL-SDR stream ended.")
            self.bytes_read += len(raw)
            self._record(raw)
            self._warn(self.source_monitor.note_bytes(raw))
            self._decode(lambda: dec.feed_raw(raw))

        self._emit_stats(status="STOPPED")

    def _run_autoset(self, dongle_header: bytes) -> None:
        from lrfhss_autoset import AutoSetCancelled, run_autoset, supported_gains
        gains = supported_gains(dongle_header)
        # Manual tuner gain and disabled digital AGC. Command parameters are
        # network-order uint32; gain itself is signed tenths of a dB.
        self.sock.sendall(struct.pack("!BI", 0x03, 1))
        self.sock.sendall(struct.pack("!BI", 0x08, 0))

        def read_iq(seconds, *, settling=False):
            target = 0 if settling else int(FS*seconds)*2
            result = bytearray()
            discarded = 0
            deadline = time.monotonic()+max(3., seconds+2.)
            settle_until = time.monotonic()+seconds
            while len(result) < target or (settling and
                    (time.monotonic() < settle_until or discarded % 2)):
                if self.stop_event.is_set():
                    raise AutoSetCancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("No fresh IQ during Auto Set; check the RTL-SDR.")
                count = (1 if time.monotonic() >= settle_until and discarded % 2 else 65536
                         ) if settling else min(65536, target-len(result))
                try:
                    block = self.sock.recv(count)
                except socket.timeout:
                    continue
                if not block:
                    raise ConnectionError("RTL-SDR stream ended during Auto Set.")
                if settling:
                    # Drain in wall time as well as RF time: queued samples
                    # from before the command must never set the next gain.
                    # recv() can split an IQ pair at any byte; always finish
                    # that pair before the next capture begins.
                    discarded += len(block)
                else:
                    result.extend(block)
            return bytes(result)

        def set_gain(gain):
            self.sock.sendall(struct.pack("!BI", 0x04, round(gain*10) & 0xffffffff))
            read_iq(.4, settling=True)

        try:
            result = run_autoset(gains, self.config.gain, self.config.trigger_db,
                                 set_gain=set_gain, capture=read_iq,
                                 cancelled=self.stop_event.is_set, progress=self.message.emit)
        except AutoSetCancelled:
            self.message.emit("Auto Set cancelled; settings kept.")
            return
        self.autoset_result.emit(result)
        self.message.emit(result["summary"])

    def _recv_exact_interruptible(self, n: int) -> bytes:
        if self.sock is None:
            return b""
        buf = bytearray()
        while len(buf) < n and not self.stop_event.is_set():
            try:
                if self.decoder is not None and self.decoder.header_idle_service and not select.select([self.sock],[],[],.002)[0]:
                    self.decoder.note_input_idle()
                    monitor = getattr(self, "rr2_health", None)
                    if monitor is not None:
                        monitor.idle()
                    self._decode(self.decoder.service_idle)
                    self._watch()
                    continue
                x = self.sock.recv(n - len(buf))
            except socket.timeout:
                self._watch()
                continue
            except OSError as exc:
                # the bytes of this block received before the failure belong to the decoder (10.4l)
                exc.partial = bytes(buf)
                raise
            if not x:
                break
            buf.extend(x)
            monitor = getattr(self, "rr2_health", None)
            if monitor is not None:
                monitor.received(len(x))
        return bytes(buf)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Read rtl_tcp's stderr for its whole life, keeping the last lines."""
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    self.stderr_tail.append(line)
        except Exception:
            pass

    def _cleanup(self) -> None:
        """Release every resource independently: a failure in one step (for example
        decoder.close()) never skips the socket, rtl_tcp or the later steps; the first
        failure is re-raised after all of them ran."""
        first_error: Exception | None = None
        if self.decoder is not None:
            try:
                self.decoder.close()
            except Exception as exc:
                first_error = exc
            try:
                self._emit_stats(status="STOPPED")
            except Exception as exc:
                first_error = first_error or exc
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.5)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1.5)
                except Exception:
                    pass
        self.proc = None
        if self.stderr_thread is not None:
            self.stderr_thread.join(timeout=1.0)
            self.stderr_thread = None
        if self.recorder is not None:
            try:
                rec = self.recorder.close()
                self.message.emit(f"Raw IQ: {rec['bytes_written']/1e6:.1f} MB saved to {rec['path']}"
                                  + (f" ({rec['bytes_dropped']} bytes dropped)" if rec['bytes_dropped'] else ""))
            except Exception as exc:
                first_error = first_error or exc
            self.recorder = None
        if first_error is not None:
            raise first_error
