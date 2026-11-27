#!/usr/bin/env python3
"""Main Fleet Manager window.

This module coordinates the receiver worker, the GUI-neutral FleetSession model,
and Qt widgets. Signal processing remains in lrfhss_live.py.
"""
from __future__ import annotations

from collections import deque
import json
import os
import time
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from lrfhss_live import (
    DEFAULT_CENTER_HZ, DEFAULT_GAIN_DB, DEFAULT_TRIGGER_DB, DEFAULT_CARRIER_SEARCH_HZ,
)

from .config import (
    APP_NAME, FLEET_RTL_TCP, LIVE_USB_BYTES, LOST_SECONDS, MAX_PACKET_TABLE_ROWS,
    SHORT_USB_BYTES, STALE_SECONDS, ReceiverConfig, default_rtl_tcp,
    HOP_SHARED, HOP_RR2, LEGACY_HOP_MODES, RR2_TRIGGER_DB,
)
from .packet_table import PacketRows
from .fleet_table import FleetTable
from .health import ReceptionEstimate, readings, validity_mask, stale
from .hop_view import HopBook
from .bridge import ReceiverBridge
from .display_io import FileJobs, JournalReader
from .gui_shutdown import QuitRelay
from . import gui_files, operator_layout
from .session import ChipSatState, FleetSession
from .theme import (
    ACCENT, APP_STYLE, BAD, BORDER, GOOD, MUTED, PANEL, PURPLE, TEXT, WARN,
)
from .widgets import MetricCard, PlotPane, ValidityPill

class FleetManager(QtWidgets.QMainWindow):
    def __init__(self, *, log_dir: Path | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1366, 768)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(APP_STYLE)
        pg.setConfigOptions(antialias=True, background=PANEL, foreground=TEXT)
        self.worker = None
        self.log_dir = Path(log_dir) if log_dir is not None else Path(__file__).resolve().parents[2] / 'logs'
        self.public_session = False
        self.active_recovery = None
        self.pending_recovery = []
        self.closing = False
        self._clear_requested = False
        self.quit_relay = QuitRelay(self)
        self.starting = False
        self.loading_settings = True
        self.session_files = []
        self.fleet_items = {}
        self.autosetting = False
        self.autoset_summary = ''
        self.session = FleetSession()
        self.reception = ReceptionEstimate()
        self.hops = HopBook()
        self.selected_csid = None
        self.log_path = None
        self.receiver_log_summary = {}
        self._hop_mode = HOP_SHARED
        self._shared_rf_settings = (DEFAULT_TRIGGER_DB, DEFAULT_CARRIER_SEARCH_HZ / 1000)
        self._two_trigger_db = RR2_TRIGGER_DB
        self._single_256_sic = True
        self._loading_hop_settings = False
        self._dirty_csids = {}
        self._pending_packets = deque(maxlen=MAX_PACKET_TABLE_ROWS)
        self._summary_dirty = True
        self._detail_dirty = False
        self._stats_dirty = False
        self._latest_stats = {}
        self._helper_visibility_path = None
        self._next_age_refresh = 0.
        self._reader_done = False
        self.file_jobs = FileJobs(self)
        self.file_jobs.idle.connect(self._files_idle)
        self.file_jobs.quiescent.connect(self._files_quiescent)
        self.file_jobs.notice.connect(self._message)
        self._build_ui()
        self.main_tabs.currentChanged.connect(self._detail_changed)
        self.detail_tabs.currentChanged.connect(self._detail_changed)
        self.run_name_edit.editingFinished.connect(self._save_gui_settings)
        self.gain_db.editingFinished.connect(self._save_gui_settings)
        self._set_receiver_running(False)
        # Four display frames per second keep operator feedback responsive while
        # batching journal deliveries and avoiding per-packet Qt/plot work.
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.setInterval(250)
        self.display_timer.timeout.connect(self._refresh_display)
        self.display_timer.start()
        self.age_timer = self.display_timer  # Existing integrations can stop refresh.
        self.file_jobs.submit(lambda: gui_files.read_state(self.log_dir), self._settings_loaded)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        operator_layout.build(self)


    def _labeled(self, text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet(f"color:{MUTED}; font-size:8pt;")
        box.addWidget(lab)
        box.addWidget(widget)
        w = QtWidgets.QWidget()
        w.setLayout(box)
        return w


    def _build_dashboard(self) -> QtWidgets.QWidget:
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_fleet_panel())
        split.addWidget(self._build_detail_panel())
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 7)
        split.setSizes([620, 710])

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.addWidget(split)
        return w

    def _build_fleet_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("CHIPSATS HEARD")
        title.setObjectName("sectionTitle")
        self.fleet_hint = QtWidgets.QLabel("Select a row for telemetry history")
        self.fleet_hint.setStyleSheet(f"color:{MUTED}; font-size:8pt;")
        hdr.addWidget(title)
        hdr.addStretch(1)
        hdr.addWidget(self.fleet_hint)
        lay.addLayout(hdr)

        self.fleet_table = FleetTable()
        self.fleet_table.setHorizontalHeaderLabels([
            "CSID", "SoC", "Counter", "Last Heard", "Packets", "Valid sensors", "GPS"
        ])
        self.fleet_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fleet_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fleet_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fleet_table.setAlternatingRowColors(True)
        self.fleet_table.verticalHeader().setVisible(False)
        self.fleet_table.verticalHeader().setDefaultSectionSize(28)
        h = self.fleet_table.horizontalHeader()
        h.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        h.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        self.fleet_table.setColumnWidth(5, 150)
        h.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeToContents)
        self.fleet_table.setSortingEnabled(True)
        self.fleet_table.sortItems(0, QtCore.Qt.AscendingOrder)
        self.fleet_table.itemSelectionChanged.connect(self._fleet_selection_changed)
        lay.addWidget(self.fleet_table, 1)
        return box

    def _build_detail_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)

        header = QtWidgets.QHBoxLayout()
        self.detail_title = QtWidgets.QLabel("NO CHIPSAT SELECTED")
        self.detail_title.setObjectName("selectedTitle")
        self.detail_last = QtWidgets.QLabel("Waiting for telemetry")
        self.detail_last.setStyleSheet(f"color:{MUTED};")
        header.addWidget(self.detail_title)
        header.addStretch(1)
        header.addWidget(self.detail_last)
        lay.addLayout(header)

        metrics = QtWidgets.QGridLayout()
        metrics.setSpacing(7)
        self.m_soc = MetricCard("Battery", "--", "state of charge")
        self.m_counter = MetricCard("Packet Counter", "--", "latest")
        self.m_temp = MetricCard("Temperature", "--", "environmental")
        self.m_alt = MetricCard("GNSS Altitude", "--", "MSL")
        self.m_cfo = MetricCard("CFO", "--", "header estimate")
        self.m_packets = MetricCard("Packets", "--", "this session")
        cards = [self.m_soc, self.m_counter, self.m_temp, self.m_alt, self.m_cfo, self.m_packets]
        for i, card in enumerate(cards):
            metrics.addWidget(card, 0, i)
        lay.addLayout(metrics)

        validity = QtWidgets.QHBoxLayout()
        validity.setSpacing(5)
        self.validity_pills: dict[str, ValidityPill] = {}
        labels = [
            ("linear_acceleration", "ACCEL"),
            ("gyroscope", "GYRO"),
            ("magnetometer", "MAG"),
            ("quaternion", "QUAT"),
            ("gps", "GPS"),
            ("state_of_charge", "POWER"),
            ("environmental", "ENV"),
            ("all_data_fresh", "FRESH"),
        ]
        for key, text in labels:
            pill = ValidityPill(text)
            self.validity_pills[key] = pill
            validity.addWidget(pill)
        lay.addLayout(validity)

        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.addTab(self._build_overview_tab(), "Overview")
        self.detail_tabs.addTab(self._build_imu_tab(), "IMU")
        self.detail_tabs.addTab(self._build_environment_tab(), "Environment")
        self.detail_tabs.addTab(self._build_gps_tab(), "GPS")
        self.detail_tabs.addTab(self._build_raw_tab(), "Raw")
        lay.addWidget(self.detail_tabs, 1)
        return box

    def _plot_pair(self, p1: PlotPane, p2: PlotPane) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(5)
        lay.addWidget(p1, 1)
        lay.addWidget(p2, 1)
        return w

    def _build_overview_tab(self):
        self.plot_soc = PlotPane("State of Charge", "%")
        self.plot_soc_curve = self.plot_soc.plot(pen=pg.mkPen(ACCENT, width=2))
        self.plot_alt = PlotPane("Altitude", "m")
        self.plot_alt_curve = self.plot_alt.plot(pen=pg.mkPen(PURPLE, width=2))
        return self._plot_pair(self.plot_soc, self.plot_alt)

    def _build_imu_tab(self):
        self.plot_accel = PlotPane("Linear Acceleration", "m/s²")
        self.plot_accel.addLegend(offset=(8, 8))
        self.accel_curves = [
            self.plot_accel.plot(pen=pg.mkPen(c, width=1.7), name=n)
            for c, n in ((ACCENT, "X"), (GOOD, "Y"), (WARN, "Z"))
        ]
        self.plot_gyro = PlotPane("Gyroscope", "deg/s")
        self.plot_gyro.addLegend(offset=(8, 8))
        self.gyro_curves = [
            self.plot_gyro.plot(pen=pg.mkPen(c, width=1.7), name=n)
            for c, n in ((ACCENT, "X"), (GOOD, "Y"), (WARN, "Z"))
        ]
        return self._plot_pair(self.plot_accel, self.plot_gyro)

    def _build_environment_tab(self):
        self.plot_temp = PlotPane("Temperature", "°C")
        self.temp_curve = self.plot_temp.plot(pen=pg.mkPen(BAD, width=2))
        self.plot_pressure = PlotPane("Pressure / Humidity", "")
        self.plot_pressure.addLegend(offset=(8, 8))
        self.press_curve = self.plot_pressure.plot(pen=pg.mkPen(ACCENT, width=2), name="Pressure hPa")
        self.hum_curve = self.plot_pressure.plot(pen=pg.mkPen(GOOD, width=2), name="Humidity %")
        return self._plot_pair(self.plot_temp, self.plot_pressure)

    def _build_gps_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(7, 7, 7, 7)
        top = QtWidgets.QHBoxLayout()
        self.gps_lat = MetricCard("Latitude", "--", "deg")
        self.gps_lon = MetricCard("Longitude", "--", "deg")
        self.gps_alt = MetricCard("Altitude", "--", "m MSL")
        top.addWidget(self.gps_lat)
        top.addWidget(self.gps_lon)
        top.addWidget(self.gps_alt)
        lay.addLayout(top)
        self.plot_gps_alt = PlotPane("GNSS Altitude History", "m")
        self.gps_alt_curve = self.plot_gps_alt.plot(pen=pg.mkPen(PURPLE, width=2))
        lay.addWidget(self.plot_gps_alt, 1)
        return w

    def _build_raw_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(7, 7, 7, 7)
        self.raw_text = QtWidgets.QPlainTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setStyleSheet(
            f"font-family:'Cascadia Mono','Consolas',monospace; background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};"
        )
        self.raw_text.setPlaceholderText("Select a ChipSat after receiving telemetry.")
        lay.addWidget(self.raw_text)
        return w

    def _build_packet_log(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(6)
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("PACKET LOG")
        title.setObjectName("sectionTitle")
        self.packet_hint = QtWidgets.QLabel("PHY failures remain visible; engineering data is shown only after PHY + application CRC checks.")
        self.packet_hint.setStyleSheet(f"color:{MUTED}; font-size:8pt;")
        hdr.addWidget(title)
        hdr.addStretch(1)
        hdr.addWidget(self.packet_hint)
        lay.addLayout(hdr)

        self.packet_table = QtWidgets.QTableView()
        self.packet_model = PacketRows(MAX_PACKET_TABLE_ROWS, self.packet_table)
        self.packet_table.setModel(self.packet_model)
        self.packet_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.packet_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.packet_table.setAlternatingRowColors(True)
        self.packet_table.verticalHeader().setVisible(False)
        self.packet_table.verticalHeader().setDefaultSectionSize(29)
        h = self.packet_table.horizontalHeader()
        examples = ("23:59:59.999", "DeSCENT_V2_5_CRC_FAIL", "65535", "65535", "100.0%", "PASS", "PASS", "−19950 Hz", "1023")
        metrics = self.packet_table.fontMetrics()
        for c, example in enumerate(examples):
            h.setSectionResizeMode(c, QtWidgets.QHeaderView.Interactive)
            label = self.packet_model.headerData(c, QtCore.Qt.Horizontal)
            h.resizeSection(c, max(metrics.horizontalAdvance(label), metrics.horizontalAdvance(example)) + 24)
        h.setSectionResizeMode(9, QtWidgets.QHeaderView.Stretch)
        h.setSectionResizeMode(10, QtWidgets.QHeaderView.ResizeToContents)
        lay.addWidget(self.packet_table, 1)
        return w

    # ---------------- receiver control ----------------
    def _load_hop_settings(self, saved):
        try:
            mode = saved.get('hop_id_mode', HOP_SHARED)
            mode = LEGACY_HOP_MODES.get(mode, mode)
            trigger = float(saved.get('shared_trigger_db', DEFAULT_TRIGGER_DB))
            carrier = float(saved.get('shared_carrier_search_khz', DEFAULT_CARRIER_SEARCH_HZ / 1000))
            two_trigger = float(saved.get('two_trigger_db', RR2_TRIGGER_DB))
            ids = (saved.get('single_id', 256), *saved.get('two_ids', (256, 299)))
            if (mode not in (HOP_SHARED, HOP_RR2) or not (2 <= trigger <= 30 and 4.5 <= carrier <= 25)
                    or not 2 <= two_trigger <= 30 or len(ids) != 3
                    or any(type(i) is not int or not 0 <= i <= 511 for i in ids)):
                return
            self._loading_hop_settings = True
            self.weak_signal_search.setChecked(saved.get('weak_signal_search') is True)
            self._shared_rf_settings = (trigger, carrier)
            self._two_trigger_db = two_trigger
            self._single_256_sic = saved.get('single_256_sic', True) is True
            for box, value in zip((self.single_id, self.first_id, self.second_id), ids):
                box.setValue(value)
            with QtCore.QSignalBlocker(self.hop_mode_combo):
                self.hop_mode_combo.setCurrentIndex(self.hop_mode_combo.findData(mode))
            self._apply_hop_mode(mode)
            if isinstance(saved.get('run_name'), str):
                self.run_name_edit.setText(saved['run_name'])
        except (ValueError, TypeError, AttributeError):
            pass  # Older settings retain the existing shared-ID defaults.
        finally:
            self._loading_hop_settings = False

    def _hop_mode_changed(self):
        if self._hop_mode == HOP_SHARED:
            self._shared_rf_settings = (self.trigger_db.value(), self.carrier_search_khz.value())
        else:
            self._two_trigger_db = self.trigger_db.value()
        self._apply_hop_mode(self.hop_mode_combo.currentData())
        self._save_gui_settings()

    def _apply_hop_mode(self, mode):
        self._hop_mode = mode
        rr2 = mode == HOP_RR2
        trigger, carrier = (self._two_trigger_db, DEFAULT_CARRIER_SEARCH_HZ / 1000) if rr2 else self._shared_rf_settings
        self.trigger_db.setValue(trigger)
        self.carrier_search_khz.setValue(carrier)
        self._update_hop_controls()

    def _selected_hop_ids(self):
        if self._hop_mode == HOP_RR2:
            return self.first_id.value(), self.second_id.value()
        return (self.single_id.value(),)

    def _sic_toggled(self, checked):
        self._single_256_sic = checked

    def _update_trigger_note(self):
        self.trigger_note.setVisible(self._hop_mode == HOP_RR2 and self.trigger_db.value() != RR2_TRIGGER_DB)

    def _update_hop_controls(self):
        two = self._hop_mode == HOP_RR2
        ids = self._selected_hop_ids()
        self.hops.configure(ids)
        supports_sic = ids == (256,)
        editable = self.hop_mode_combo.isEnabled()
        self.single_id.setVisible(not two)
        self.first_id.setVisible(two)
        self.second_id.setVisible(two)
        self.hop_note.setVisible(tuple(sorted(ids)) not in ((256,), (256, 299)))
        with QtCore.QSignalBlocker(self.sic):
            self.sic.setChecked(supports_sic and self._single_256_sic)
        self.sic.setEnabled(editable and supports_sic)
        self.autoset_btn.setEnabled(editable and supports_sic and self.source_combo.currentData() == "live")
        self.trigger_db.setEnabled(editable)
        self.carrier_search_khz.setEnabled(editable and not two)
        self.carrier_search_khz.setToolTip(
            'Two-ID acquisition supports ±4.5 kHz only.' if two else
            'Search half-width around each expected header carrier; wider settings increase CPU use.')
        self._update_trigger_note()

    def _source_changed(self):
        self._helper_visibility_path = None
        file_mode = self.source_combo.currentData() == "file"
        running = self.worker is not None or self.starting or self.loading_settings
        self.file_label.setVisible(file_mode)
        self.file_edit.setVisible(file_mode)
        self.browse_btn.setVisible(file_mode)
        self.exhaustive_file.setVisible(file_mode)
        self.exhaustive_file.setEnabled(file_mode and not running)
        show_usb = not file_mode and default_rtl_tcp() == str(FLEET_RTL_TCP)
        self.usb_label.setVisible(show_usb)
        self.usb_transfer.setVisible(show_usb)
        self.usb_transfer.setEnabled(show_usb and not running)
        self.file_edit.setEnabled(file_mode and not running)
        self.browse_btn.setEnabled(file_mode and not running)
        self.gain_db.setEnabled((not file_mode) and not running)
        self.ppm.setEnabled((not file_mode) and not running)
        self._update_hop_controls()

    def _browse_iq(self):
        dialog = QtWidgets.QFileDialog(self, 'Open RTL-SDR IQ Capture')
        dialog.setNameFilter('IQ files (*.iq *.bin *.raw);;All files (*.*)')
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.fileSelected.connect(self.file_edit.setText)
        dialog.show()

    def _toggle_receiver(self):
        if self.worker is not None:
            self._stop_receiver()
        else:
            self._start_receiver()

    def _start_autoset(self):
        if self.autoset_btn.isEnabled():
            self._start_receiver(autoset_only=True)

    def _start_receiver(self, *, autoset_only=False):
        if (self.worker is not None or self.starting or self.loading_settings
                or self.active_recovery is not None or self.pending_recovery):
            return
        try:
            config = self._receiver_config(autoset_only=autoset_only)
        except ValueError as exc:
            self.status.showMessage(str(exc))
            return
        self.autosetting = autoset_only
        self.autoset_summary = ''
        self.starting = True
        self._set_receiver_running(True)
        self.start_btn.setEnabled(False)
        public = self.auto_log.isChecked() and not autoset_only
        name = self.run_name_edit.text().strip()
        self._save_gui_settings()
        self.file_jobs.submit(
            lambda: gui_files.prepare_session(self.log_dir, config, public, name),
            lambda result, error: self._session_prepared(config, public, result, error))

    def _delivery_event(self, kind, value):
        handlers = {"status": self._on_receiver_stats, "message": self._message,
                    "error": self._receiver_error, "autoset_result": self._on_autoset_result}
        if kind in handlers:
            handlers[kind](value)

    def _recover(self):
        if self.active_recovery is not None:
            return
        if not self.pending_recovery or self.closing:
            self._set_receiver_running(False)
            return
        journal = self.pending_recovery.pop(0)
        self.starting = True
        self._set_receiver_running(False)
        self.file_jobs.submit(lambda: gui_files.recovery_metadata(journal),
                              lambda result, error: self._recovery_prepared(journal, result, error))

    def _on_packet_batch(self, envelopes):
        # Accept every packet into the unchanged session model. Rendering is
        # bounded by the display timer, not the packet publication rate.
        for envelope in envelopes:
            self.hops.add(envelope)
            update = self.session.add_packet(envelope, ignore_csid_zero=self.ignore_zero.isChecked())
            if update.fleet_updated:
                self.reception.add(update.csid, envelope['record']['decoded']['packet_counter'])
                self._dirty_csids[update.csid] = None
                if self.selected_csid is None:
                    self.selected_csid = update.csid
                if self.selected_csid == update.csid:
                    self._detail_dirty = True
        self._pending_packets.extend(envelopes)
        self._summary_dirty = True

    def _append_packet_rows(self, envelopes):
        self.packet_model.insert_packets([self._packet_values(env) for env in envelopes])

    def _refresh_packet_card(self):
        self.card_packets.set_value(str(self.session.total_phy_ok))
        self.card_failed.set_value(str(self.session.total_phy_fail), color=BAD if self.session.total_phy_fail else None)
        retry = self.session.receiver_stages.get('adaptive_retry')
        pending = retry['waiting'] + retry['active'] if retry is not None else None
        self.card_packets.setToolTip('CRC-valid PHY packets' +
            (f'; {pending} pending adaptive retries' if pending is not None else ''))

    def _stop_receiver(self):
        if self.worker is not None:
            self.status.showMessage("Cancelling Auto Set..." if self.autosetting else "Stopping receiver...")
            self.worker.request_stop()
        else:
            self.file_jobs.quiesce()
        self.start_btn.setEnabled(False)

    def _receiver_finished(self):
        was_autoset = self.autosetting
        self.autosetting = False
        self.starting = False
        self._refresh_display()
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self._set_receiver_running(False)
        self._close_session_log()
        self.file_jobs.quiesce()
        self.status.showMessage((self.autoset_summary or 'Auto Set stopped; settings kept.')
                                if was_autoset else 'Receiver stopped.')
        if self.closing:
            QtCore.QTimer.singleShot(0, self.close)

    @QtCore.Slot(object)
    def _on_autoset_result(self, result):
        # The result can already be queued when Cancel is clicked on the GUI
        # thread. Keep the old controls in that case, too.
        if not self.autosetting or self.worker is None or self.worker.stop_event.is_set():
            return
        self.gain_db.setValue(result["gain_db"])
        self.trigger_db.setValue(result["trigger_db"])
        self.autoset_summary = result["summary"]

    def _set_receiver_running(self, running: bool):
        waiting = self.loading_settings or self.starting or self.active_recovery is not None or bool(self.pending_recovery)
        self.start_btn.setEnabled(not waiting)
        self.start_btn.setText('Cancel Auto Set' if running and self.autosetting
                               else 'Stop receiver' if running else 'Start receiver')
        self.start_btn.setObjectName('danger' if running else 'primary')
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        for widget in (self.run_name_edit, self.hop_mode_combo, self.single_id, self.first_id, self.second_id,
                       self.source_combo, self.freq_mhz,
                       self.gain_db, self.ppm, self.trigger_db, self.carrier_search_khz,
                       self.file_edit, self.browse_btn, self.record_iq, self.weak_signal_search):
            widget.setEnabled(not running and not waiting)
        self._source_changed()
        if running:
            self.card_receiver.set_value('AUTO SET' if self.autosetting else 'STARTING', 'Opening source', WARN)
        else:
            self.card_receiver.set_value('OFFLINE', 'Not connected', MUTED)
        self._set_listening_state(running)

    @QtCore.Slot(object)
    def _on_receiver_stats(self, stats: dict):
        if self.public_session:
            self.receiver_log_summary['last_stats'] = stats
        if 'receiver_stages' in stats:
            self.session.update_receiver_stats(stats['receiver_stages'])
            self._summary_dirty = True
        self._latest_stats.update(stats)
        self._stats_dirty = True
        # Packet totals are session-wide and are updated in _on_packet().
        # Receiver statistics here are source/run diagnostics only.

    @QtCore.Slot(str)
    def _message(self, text: str):
        self.status.showMessage(text, 10000)

    @QtCore.Slot(str)
    def _receiver_error(self, text: str):
        if self.autosetting:
            self.autoset_summary = text
        self.status.showMessage(text)
        self.card_receiver.set_value('ERROR', text, BAD)
        self.receiver_detail.setText(text)

    # ---------------- packet model ----------------
    @QtCore.Slot(object)
    def _on_packet(self, env: dict[str, Any]):
        self._on_packet_batch([env])

    def _packet_values(self, env):
        r = env.get("record", {})
        ptype = r.get("packet_type", "UNKNOWN")
        phy_ok = bool(env.get("phy", {}).get("crc_ok", False))
        decoded = r.get("decoded", {}) if ptype == "DeSCENT_V2_5" else {}
        app = "PASS" if ptype == "DeSCENT_V2_5" else ("FAIL" if ptype == "DeSCENT_V2_5_CRC_FAIL" else "--")
        cfo = env.get("phy", {}).get("cfo_hz")
        values = [
            self._fmt_time(env.get("received_at", "")),
            ptype,
            str(decoded.get("csid", "--")),
            str(decoded.get("packet_counter", "--")),
            f"{decoded.get('state_of_charge_percent'):.1f}%" if decoded else "--",
            "PASS" if phy_ok else "FAIL",
            app,
            f"{cfo:+.0f} Hz" if isinstance(cfo, (int, float)) else "--",
            str(r.get("raw_length_bytes", "--")),
            r.get("decode_error", r.get("raw_hex", "")),
            str(env.get("phy", {}).get("hop_seq_id", "--")),
        ]
        return values

    def _fmt_time(self, iso: str) -> str:
        try:
            return datetime.fromisoformat(iso).strftime("%H:%M:%S.%f")[:-3]
        except Exception:
            return str(iso)

    def _upsert_fleet_row(self, csid: int):
        state = self.session.fleet[csid]
        d = state.latest
        age = max(0., time.time()-state.last_seen)
        mask = validity_mask(d)
        values = (str(csid), f"{d['state_of_charge_percent']:.1f}%", str(d['packet_counter']),
                  self._age_label(age), str(state.count), '', 'VALID' if d.get('valid', {}).get('gps') else '--')
        sorts = (csid, d['state_of_charge_percent'], d['packet_counter'], int(age), state.count, mask, values[6])
        self.fleet_items[csid] = self.fleet_table.upsert(csid, values, sorts, mask, age >= STALE_SECONDS)

    def _row_for_csid(self, csid: int) -> int:
        item = self.fleet_items.get(csid)
        return self.fleet_table.row(item) if item is not None else -1

    def _fleet_selection_changed(self):
        rows = self.fleet_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.fleet_table.item(rows[0].row(), 0)
        if item:
            self._select_csid(int(item.data(QtCore.Qt.UserRole)))

    def _select_csid(self, csid: int):
        if csid not in self.session.fleet:
            return
        self.selected_csid = csid
        row = self._row_for_csid(csid)
        if row >= 0 and not self.fleet_table.item(row, 0).isSelected():
            self.fleet_table.selectRow(row)
        self._detail_dirty = True

    # ---------------- detail / plots ----------------
    def _refresh_detail(self, csid: int):
        state = self.session.fleet.get(csid)
        if not state or state.latest is None:
            return
        d = state.latest
        env = state.envelope
        phy = env.get("phy", {})
        age = time.time() - state.last_seen

        self.detail_title.setText(f"CHIPSAT {csid}")
        self.detail_last.setText(f"Last heard {self._age_label(age)} ago  •  {state.count} packets")

        soc = d["state_of_charge_percent"]
        soc_color = GOOD if soc >= 35 else WARN if soc >= 20 else BAD
        self.m_soc.set_value(f"{soc:.1f}%", "state of charge", soc_color)
        self.m_counter.set_value(str(d["packet_counter"]), "latest")
        self.m_temp.set_value(f"{d['temperature_C']:.1f} °C", "environmental")
        self.m_alt.set_value(f"{d['gps_altitude_msl_m']:.1f} m", "MSL")
        cfo = phy.get("cfo_hz")
        self.m_cfo.set_value(f"{cfo:+.0f} Hz" if isinstance(cfo, (int, float)) else "--", "header estimate")
        self.m_packets.set_value(str(state.count), "this session")

        for k, pill in self.validity_pills.items():
            pill.set_state(d.get("valid", {}).get(k))

        self.gps_lat.set_value(f"{d['latitude_deg']:.7f}", "deg")
        self.gps_lon.set_value(f"{d['longitude_deg']:.7f}", "deg")
        self.gps_alt.set_value(f"{d['gps_altitude_msl_m']:.1f}", "m MSL")

        if self.detail_tabs.currentIndex() == 4:
            self.raw_text.setPlainText(self._raw_detail_text(env))
        self._refresh_plots(csid)

    def _raw_detail_text(self, env: dict[str, Any]) -> str:
        r = env.get("record", {})
        d = r.get("decoded", {})
        phy = env.get("phy", {})
        retry = phy.get("payload_retry", {})
        retry_lines = []
        if retry.get("accepted") is not None:
            dt, df = retry["accepted"]
            retry_lines.append(f"Adaptive recovery: {retry['trials']} trial(s), {dt:+d} samples, {df:+g} Hz")
        pieces = [
            f"Received: {env.get('received_at', '--')}",
            f"Center frequency: {env.get('center_frequency_hz', 0)/1e6:.6f} MHz",
            f"LR-FHSS PHY CRC: {'PASS' if phy.get('crc_ok') else 'FAIL'}",
            f"Fragments: {phy.get('fragments', '--')}",
            ("Spectral quality: not measured (raw CRC passed)"
             if phy.get('crc_ok') and phy.get('payload_method') == 'raw'
             and (phy.get('fragment_quality') or {}).get('spectral_quality_measured') is False
             else "Spectral quality: " + json.dumps(phy.get('fragment_quality'))),
            f"Hop sequence: 0x{int(phy.get('hop_seq_id', 0)):03X}",
            f"CFO estimate: {phy.get('cfo_hz', '--')} Hz",
            f"Header Viterbi cost: {phy.get('header_cost', '--')}",
            f"Payload Viterbi cost: {phy.get('viterbi_cost', '--')}",
            *retry_lines,
            "",
            "DECODED RECORD",
            json.dumps(d, indent=2),
            "",
            "RAW PACKET",
            r.get("raw_hex", ""),
        ]
        return "\n".join(pieces)

    def _refresh_plots(self, csid: int):
        if not self.isVisible() or self.isMinimized() or self.main_tabs.currentIndex() != 0:
            return
        tab = self.detail_tabs.currentIndex()
        if tab == 4:
            return
        hist = list(self.session.histories.get(csid, []))
        x = list(range(len(hist)))
        data = [entry['d'] for entry in hist]
        if tab == 0:
            self.plot_soc_curve.setData(x, [d['state_of_charge_percent'] for d in data])
            self.plot_alt_curve.setData(x, [d['environmental_altitude_m'] for d in data])
        elif tab == 1:
            for axis in range(3):
                self.accel_curves[axis].setData(x, [d['linear_acceleration_mps2'][axis] for d in data])
                self.gyro_curves[axis].setData(x, [d['gyroscope_deg_s'][axis] for d in data])
        elif tab == 2:
            self.temp_curve.setData(x, [d['temperature_C'] for d in data])
            self.press_curve.setData(x, [d['pressure_hPa'] for d in data])
            self.hum_curve.setData(x, [d['humidity_percent'] for d in data])
        elif tab == 3:
            self.gps_alt_curve.setData(x, [d['gps_altitude_msl_m'] for d in data])

    # ---------------- summaries / aging ----------------
    def _refresh_ages(self):
        now = time.time()
        for csid, state in self.session.fleet.items():
            before = len(self.session.histories.get(csid, ()))
            self.session.prune_history(csid, now)
            if csid == self.selected_csid and len(self.session.histories.get(csid, ())) != before:
                self._detail_dirty = True
            row = self._row_for_csid(csid)
            if row >= 0:
                self._update_row_age(row, state)
        if self.selected_csid in self.session.fleet:
            s = self.session.fleet[self.selected_csid]
            self.detail_last.setText(
                f"Last heard {self._age_label(now-s.last_seen)} ago  •  {s.count} packets"
            )
        self._update_summary()

    def _update_row_age(self, row: int, state: ChipSatState):
        age = max(0.0, time.time() - state.last_seen)
        self.fleet_table.age(state.csid, self._age_label(age), age, stale(state.last_seen, time.time(), STALE_SECONDS))

    def _age_label(self, age: float) -> str:
        if age < 1:
            return "<1 s"
        if age < 60:
            return f"{int(age)} s"
        if age < 3600:
            return f"{int(age//60)}m {int(age%60):02d}s"
        return f"{int(age//3600)}h {int((age%3600)//60):02d}m"

    def _update_summary(self):
        total = len(self.session.fleet)
        active = self.session.active_count()
        # V2.5 sensor bits 0–6, as in LoRa reporting_valid_data. Freshness alone
        # is not a sensor field. Latest trusted packet, session-wide, no timeout.
        valid = sum(any(value for key, value in state.latest.get('valid', {}).items()
                        if key != 'all_data_fresh') for state in self.session.fleet.values())
        self.card_active.set_value(str(total), f'{active} active (last {STALE_SECONDS:g} s)')
        self.card_valid.set_value(str(valid))
        self.card_rate.set_value(f'{self.session.crc_valid_rate_hz():.2f} Hz', 'CRC-valid, last 60 s')
        self._render_health()

    # ---------------- logging / export ----------------
    def _close_session_log(self):
        if not self.public_session or self.log_path is None:
            return
        path, summary = self.log_path.with_suffix('.receiver.json'), dict(self.receiver_log_summary)
        self.public_session = False
        self.file_jobs.submit(lambda: gui_files.finish_summary_if_missing(path, summary), self._file_result)

    def _export_csv(self):
        dialog = QtWidgets.QFileDialog(self, 'Export Telemetry CSV')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('CSV files (*.csv)')
        dialog.setDefaultSuffix('csv')
        dialog.selectFile(f'lrfhss_telemetry_{datetime.now():%Y%m%d_%H%M%S}.csv')
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.fileSelected.connect(self._export_to_path)
        dialog.show()

    def _clear_session(self):
        if (self.worker is not None or self.starting or self.loading_settings
                or self.active_recovery is not None or self.pending_recovery):
            self.status.showMessage("Stop the receiver before clearing the session.")
            return
        if not self.file_jobs.is_quiescent:
            self._clear_requested = True
            self.status.showMessage('Finishing GUI output before clearing the session...')
            self.file_jobs.quiesce()
            return
        self._dirty_csids.clear()
        self._pending_packets.clear()
        self._latest_stats.clear()
        self._summary_dirty = True
        self.session.clear()
        self.reception = ReceptionEstimate()
        self.hops = HopBook(self._selected_hop_ids())
        self.hop_view.book = self.hops
        self.hop_view._drawn = None
        self.selected_csid = None
        self.fleet_items.clear()
        self.fleet_table.clear_rows()
        self.packet_model.clear()
        self.session_files.clear()
        self.detail_title.setText("NO CHIPSAT SELECTED")
        self.detail_last.setText("Waiting for telemetry")
        for c in (self.m_soc, self.m_counter, self.m_temp, self.m_alt, self.m_cfo, self.m_packets,
                  self.gps_lat, self.gps_lon, self.gps_alt):
            c.set_value("--")
        for pill in self.validity_pills.values():
            pill.set_state(None)
        self.raw_text.clear()
        for curve in [
            self.plot_soc_curve, self.plot_alt_curve, self.gps_alt_curve, self.temp_curve,
            self.press_curve, self.hum_curve, *self.accel_curves, *self.gyro_curves,
        ]:
            curve.setData([], [])
        self.card_active.set_value("0", f"0 active (last {STALE_SECONDS:g} s)")
        self.card_valid.set_value("0")
        self.card_failed.set_value("0")
        self.card_packets.set_value("0", "0 failed")
        self.card_app.set_value("0 pass / 0 fail", "0 failures")
        self.card_rate.set_value("0.00 Hz", "last 60 s")
        self.status.showMessage("Session cleared.", 5000)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if not self.closing and not self.loading_settings:
            self._save_gui_settings()
        self.closing = True
        if self.worker is not None:
            self.worker.request_stop()
            event.ignore()
            return
        if self.active_recovery is not None:
            self.active_recovery.cancel.set()
            event.ignore()
            return
        self._close_session_log()
        if not self.file_jobs.close():
            event.ignore()
            return
        self.display_timer.stop()
        event.accept()
        self.quit_relay.window_closed()


    def _show_setup(self):
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def _settings_loaded(self, result, error):
        if error:
            self.status.showMessage(f'Could not load GUI settings: {error}')
        else:
            saved, self.pending_recovery = result
            self._load_hop_settings(saved)
            gain = saved.get('gain_db', saved.get('gain'))
            if isinstance(gain, (int, float)) and self.gain_db.minimum() <= gain <= self.gain_db.maximum():
                self.gain_db.setValue(gain)
        self.loading_settings = False
        self._set_receiver_running(False)
        if self.pending_recovery:
            self._recover()

    def _save_gui_settings(self):
        if self.closing or self._loading_hop_settings or self.loading_settings:
            return
        shared = ((self.trigger_db.value(), self.carrier_search_khz.value())
                  if self._hop_mode == HOP_SHARED else self._shared_rf_settings)
        settings = dict(hop_id_mode=self._hop_mode, shared_trigger_db=shared[0],
                        shared_carrier_search_khz=shared[1], run_name=self.run_name_edit.text(),
                        single_id=self.single_id.value(), two_ids=[self.first_id.value(), self.second_id.value()],
                        single_256_sic=self._single_256_sic,
                        two_trigger_db=self.trigger_db.value() if self._hop_mode == HOP_RR2 else self._two_trigger_db,
                        weak_signal_search=self.weak_signal_search.isChecked(), gain_db=self.gain_db.value())
        self.file_jobs.submit(lambda: gui_files.save_settings(self.log_dir, settings), self._file_result)

    def _file_result(self, result, error):
        if error:
            self.status.showMessage(f'GUI file operation failed: {error}')

    def _files_idle(self):
        if self.closing:
            QtCore.QTimer.singleShot(0, self.close)

    def _files_quiescent(self):
        if self.closing:
            QtCore.QTimer.singleShot(0, self.close)
        elif self._clear_requested:
            self._clear_requested = False
            self._clear_session()

    def _receiver_config(self, *, autoset_only=False):
        mode = self.source_combo.currentData()
        return ReceiverConfig(
            mode=mode, hop_id_mode=self.hop_mode_combo.currentData(),
            allowed_hop_ids=self._selected_hop_ids(), phdr_sic="on" if self.sic.isChecked() else "off",
            freq=int(round(self.freq_mhz.value() * 1e6)), gain=float(self.gain_db.value()),
            ppm=int(self.ppm.value()), trigger_db=float(self.trigger_db.value()),
            carrier_search_hz=float(self.carrier_search_khz.value()) * 1000,
            weak_signal_search=self.weak_signal_search.isChecked(),
            file_path=self.file_edit.text().strip(),
            exhaustive_file=(mode == 'file' and self.exhaustive_file.isChecked()),
            record_iq=(mode == 'live' and self.record_iq.isChecked()), autoset_only=autoset_only,
            usb_transfer_bytes=SHORT_USB_BYTES if autoset_only else self.usb_transfer.currentData())

    def _session_prepared(self, config, public, result, error):
        self.starting = False
        if error:
            self._set_receiver_running(False)
            self._receiver_error(f'Could not start the receiver: {error}')
            return
        journal, self.receiver_log_summary = result
        self.session_files.append(journal.packets)
        self.public_session = public
        self.log_path = journal.packets if public else None
        if self.closing:
            self._close_session_log()
            return
        try:
            self.worker = ReceiverBridge(config, journal, self._on_packet_batch, self._delivery_event, self)
            self.worker.finished.connect(self._receiver_finished)
            self._latest_stats.clear()
            self._set_receiver_running(True)
            self.worker.start()
            if public:
                self.status.showMessage(f'Logging to {journal.packets}', 10000)
        except (OSError, RuntimeError) as exc:
            self._receiver_finished()
            self._receiver_error(f'Could not start the receiver: {exc}')

    def _recovery_prepared(self, journal, result, error):
        self.starting = False
        if error:
            self._receiver_error(f'Could not recover {journal.packets}: {error}')
            QtCore.QTimer.singleShot(0, self._recover)
            return
        reader = JournalReader(journal, complete_probe=lambda: gui_files.writer_finished(journal), parent=self)
        self.active_recovery = reader
        reader.opened.connect(lambda: self._recovery_opened(journal, result), QtCore.Qt.ConnectionType.QueuedConnection)
        reader.packet_batch.connect(self._recovery_batch)
        reader.event_batch.connect(self._recovery_events)
        reader.fault.connect(self._receiver_error)
        reader.finished.connect(self._recovery_finished)
        reader.start()

    def _recovery_opened(self, journal, metadata):
        self.public_session, self.receiver_log_summary = metadata
        self.session_files.append(journal.packets)
        self.log_path = journal.packets if self.public_session else None

    @QtCore.Slot(object, object)
    def _recovery_batch(self, rows, receipt):
        try:
            self._on_packet_batch(rows)
        except Exception as exc:
            receipt.error = str(exc)
        finally:
            receipt.accepted.set()

    @QtCore.Slot(object, object)
    def _recovery_events(self, events, receipt):
        try:
            for kind, value in events:
                self._delivery_event(kind, value)
        except Exception as exc:
            receipt.error = str(exc)
        finally:
            receipt.accepted.set()

    def _recovery_finished(self):
        if not self.active_recovery.wait(0):
            QtCore.QTimer.singleShot(20, self._recovery_finished)
            return
        self._refresh_display()
        self._close_session_log()
        self.active_recovery.deleteLater()
        self.active_recovery = None
        QtCore.QTimer.singleShot(0, self.close if self.closing else self._recover)

    def _export_to_path(self, path):
        paths = tuple(self.session_files)
        fallback = tuple(self.session.packet_records) if not paths else ()
        self.export_btn.setEnabled(False)
        def finished(count, error):
            self.export_btn.setEnabled(True)
            self.status.showMessage(f'Export failed: {error}' if error else
                (f'Exported {count} packet(s) to {path}' if count else
                 'No CRC-valid DeSCENT telemetry is available to export yet.'), 10000)
        cancel = threading.Event()
        self.file_jobs.submit(lambda: gui_files.export_csv(path, paths, fallback, cancel),
                              finished, label='CSV export', cancel=cancel)

    def _detail_changed(self, *_):
        self._detail_dirty = True

    def _refresh_display(self):
        if self._pending_packets:
            self._append_packet_rows(tuple(self._pending_packets))
            self._pending_packets.clear()
        if self._dirty_csids:
            with self.fleet_table.batch():
                for csid in self._dirty_csids:
                    self._upsert_fleet_row(csid)
        self._dirty_csids.clear()
        if self._summary_dirty:
            self._refresh_packet_card()
            self.card_app.set_value(f'{self.session.app_crc_pass} pass / {self.session.app_crc_fail} fail', f'{self.session.app_crc_fail} failures',
                                    GOOD if self.session.app_crc_pass else TEXT)
            self._update_summary()
            self._summary_dirty = False
        if self._stats_dirty:
            self._render_receiver_stats()
            self._stats_dirty = False
        if time.monotonic() >= self._next_age_refresh:
            with self.fleet_table.batch():
                self._refresh_ages()
            self._next_age_refresh = time.monotonic() + 1.
        if (self._detail_dirty and self.isVisible() and not self.isMinimized()
                and self.main_tabs.currentIndex() == 0 and self.selected_csid is not None):
            self._select_csid(self.selected_csid)
            self._refresh_detail(self.selected_csid)
            self._detail_dirty = False
        self.hop_view.maybe_refresh()

    def _set_listening_state(self, running):
        if not running:
            label, color = 'Not listening', '#ffaaa1'
        elif self.source_combo.currentData() == 'live':
            label, color = 'Listening', '#91ebc8'
        elif self.exhaustive_file.isChecked():
            label, color = 'Exhaustive replay', '#91ebc8'
        else:
            label, color = 'Replaying file', '#91ebc8'
        self.receiver_state.setText('●  ' + label)
        self.receiver_state.setStyleSheet(f'color:{color};font-size:10pt;')

    def _render_receiver_stats(self):
        s = self._latest_stats
        self._render_health()
        if s.get('status', 'RUNNING') != 'RUNNING':
            return
        helper = s.get('rtl_tcp')
        if helper and helper != self._helper_visibility_path:
            self._helper_visibility_path = helper
            self.file_jobs.submit(lambda: gui_files.is_fleet_helper(helper),
                lambda result, error: self._helper_visibility_ready(helper, result, error))
        sub = f"{s.get('bytes_read', 0)/1e6:.1f} MB IQ received"
        if 'file_progress' in s:
            sub = f"{100*s['file_progress']:.1f}% of IQ file"
            exhaustive = s.get('exhaustive_file')
            if exhaustive is not None:
                eta = exhaustive.get('eta_s')
                sub += (' | exhaustive | ETA estimating' if eta is None else
                        f' | exhaustive | ETA {int(eta)//60}:{int(eta)%60:02d}')
        retry = self.session.receiver_stages.get('adaptive_retry')
        if retry is not None:
            sub += f" | {retry['waiting'] + retry['active']} pending retries"
        rejected = (s.get('rr2_health') or {}).get('job_rejections', 0)
        if rejected:
            sub += f" | {rejected:,} jobs rejected"
        self.receiver_detail.setText(sub)
        source = s.get('source') or {}
        if source.get('stalled'):
            self.card_receiver.set_value('NO DATA', 'Check dongle', BAD)
        elif source.get('level_state') == 'clipping':
            self.card_receiver.set_value('ONLINE', sub, WARN)
        elif source.get('level_state') == 'low':
            self.card_receiver.set_value('ONLINE', sub, WARN)
        else:
            self.card_receiver.set_value('ONLINE', sub, WARN if rejected else GOOD)

    def _render_health(self):
        colors = {'normal': GOOD, 'amber': WARN, 'red': BAD}
        cells = {'gain': self.health_gain, 'adc': self.health_clipping, 'lag': self.health_lag,
                 'expired': self.health_expired, 'source': self.health_source,
                 'recording': self.health_recording, 'prr': self.health_prr}
        for key, value in readings(self._latest_stats, self.gain_db.value(), self.reception).items():
            color = MUTED if value.text.startswith('n/a') and value.level == 'normal' else colors[value.level]
            cells[key].set_value(value.text, value.tip, color)

    def _helper_visibility_ready(self, helper, result, error):
        if error or helper != self._helper_visibility_path:
            return
        show_usb = result and self.source_combo.currentData() == 'live'
        self.usb_label.setVisible(show_usb)
        self.usb_transfer.setVisible(show_usb)
