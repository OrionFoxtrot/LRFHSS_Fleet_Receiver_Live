#!/usr/bin/env python3
"""Launch the DeSCENT LR-FHSS Fleet Manager.

All GUI implementation lives in ``lrfhss_gui_app``.  Keeping this file small
makes the program entry point obvious and keeps imports stable for launchers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def main() -> int:
    from PySide6 import QtWidgets
    from lrfhss_gui_app.config import APP_NAME
    from lrfhss_gui_app.window import FleetManager
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    window = FleetManager()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
