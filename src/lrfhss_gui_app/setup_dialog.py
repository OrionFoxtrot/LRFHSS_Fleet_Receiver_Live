"""Receiver setup presents the existing controls; it owns no RF configuration."""
from PySide6 import QtCore, QtWidgets


class ReceiverSetup(QtWidgets.QDialog):
    """Modeless settings panel using the window's original widgets.

    Reusing the controls avoids a second set of RF defaults or conversion rules.
    Closing the dialog keeps edits, just as editing the old inline controls did.
    """
    def __init__(self, owner):
        super().__init__(owner)
        self.setWindowTitle("Receiver Setup")
        self.setModal(False)
        self.setMinimumWidth(520)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        heading = QtWidgets.QLabel("Receiver Setup")
        heading.setObjectName("sectionTitle")
        outer.addWidget(heading)
        description = QtWidgets.QLabel("RF and logging settings apply when the next receiver run starts.")
        description.setWordWrap(True)
        description.setObjectName("runNote")
        outer.addWidget(description)
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Center MHz", owner.freq_mhz)
        form.addRow("PPM", owner.ppm)
        trigger = QtWidgets.QVBoxLayout()
        trigger.addWidget(owner.trigger_db)
        trigger.addWidget(owner.trigger_note)
        form.addRow("Trigger dB", trigger)
        form.addRow("Carrier search ±kHz", owner.carrier_search_khz)
        form.addRow(owner.usb_label, owner.usb_transfer)
        outer.addLayout(form)
        outer.addWidget(owner.sic)
        outer.addWidget(owner.weak_signal_search)
        outer.addWidget(owner.auto_log)
        outer.addWidget(owner.ignore_zero)
        outer.addSpacing(4)
        outer.addWidget(owner.autoset_btn)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        outer.addWidget(buttons)
        owner.autoset_btn.clicked.connect(self.close)
