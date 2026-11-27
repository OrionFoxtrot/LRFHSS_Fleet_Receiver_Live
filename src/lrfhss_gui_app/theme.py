"""Qt implementation of the DeSCENT LoRa operator palette and typography.

Only the visual constants are shared by design. No LoRa repository is imported.
"""
from PySide6 import QtGui

BG = PAPER = "#edf2f3"
PANEL = "#ffffff"
PANEL_2 = "#f4f7f7"
NAVY = "#132f38"
TEXT = INK = "#17343e"
MUTED = "#45616a"
BORDER = EDGE = "#cbd7d9"
ACCENT = TEAL = "#006d60"
MINT = "#cdf3e8"
GOOD = TEAL
WARN = "#935b16"
BAD = RED = "#a73525"
PURPLE = "#635d9b"
FONT_FAMILIES = ("Inter", "Roboto", "Segoe UI", "Arial")
MONO_FAMILIES = ("Consolas", "Courier New")


def font_family(mono=False):
    installed = set(QtGui.QFontDatabase.families())
    choices = MONO_FAMILIES if mono else FONT_FAMILIES
    return next((name for name in choices if name in installed), choices[-1])


def operator_font(points=10, *, mono=False, bold=False):
    font = QtGui.QFont(font_family(mono), points)
    font.setBold(bold)
    return font


APP_STYLE = f"""
QWidget {{ color:{TEXT}; font-family:'Inter','Roboto','Segoe UI','Arial'; font-size:10pt; }}
QMainWindow, QDialog, QWidget#page {{ background:{PAPER}; }}
QLabel {{ background:transparent; }}
QFrame#panel, QFrame#runBar, QFrame#healthStrip, QGroupBox {{
    background:{PANEL}; border:1px solid {EDGE}; border-radius:0;
}}
QFrame#headline {{ background:{NAVY}; border:0; }}
QFrame#headline QLabel {{ color:#f6fbfa; }}
QLabel#overline {{ color:{MUTED}; font-size:9pt; font-weight:600; }}
QLabel#sectionTitle {{ color:{INK}; font-size:13pt; font-weight:700; }}
QLabel#selectedTitle {{ color:{INK}; font-size:19pt; font-weight:700; }}
QLabel#runNote {{ color:{MUTED}; font-size:9pt; }}
QGroupBox {{ margin-top:12px; padding:12px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; font-weight:600; }}
QPushButton {{ background:#e7efee; border:1px solid {EDGE}; padding:5px 10px; font-weight:600; min-height:20px; }}
QPushButton:hover {{ background:#dae8e5; border-color:{TEAL}; }}
QPushButton:pressed {{ background:#cee3dc; }}
QPushButton:disabled {{ color:#70828a; background:#f0f3f3; }}
QPushButton#primary {{ background:{TEAL}; color:white; border-color:{TEAL}; }}
QPushButton#primary:hover {{ background:#005a50; }}
QPushButton#danger {{ background:#29454e; color:white; border-color:#29454e; }}
QPushButton#danger:hover {{ background:#375a63; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background:white; color:{INK}; border:1px solid {EDGE}; border-radius:0;
    padding:4px 6px; min-height:20px; selection-background-color:{MINT}; selection-color:{INK};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background:#f0f3f3; color:#697e83;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color:{TEAL}; }}
QComboBox QAbstractItemView {{ background:white; color:{INK}; selection-background-color:{MINT}; selection-color:{INK}; }}
QTableView, QTableWidget {{ background:white; alternate-background-color:{PANEL_2};
    border:1px solid {EDGE}; gridline-color:#e5edee;
    selection-background-color:#d6ece6; selection-color:{INK}; }}
QHeaderView::section {{ background:#eaf0f1; color:{MUTED}; border:0; border-bottom:1px solid {EDGE};
    padding:5px 4px; font-size:9pt; font-weight:600; }}
QTabWidget::pane {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:0; }}
QTabBar::tab {{ background:#e8eef0; color:{INK}; border:1px solid {EDGE}; padding:6px 12px;
    margin-right:2px; font-weight:600; }}
QTabBar::tab:selected {{ background:{NAVY}; color:white; }}
QTabBar::tab:hover:!selected {{ background:#dce9e6; }}
QScrollBar:vertical {{ background:#edf2f3; width:12px; margin:0; }}
QScrollBar::handle:vertical {{ background:#aebfc2; min-height:24px; }}
QScrollBar:horizontal {{ background:#edf2f3; height:12px; margin:0; }}
QScrollBar::handle:horizontal {{ background:#aebfc2; min-width:24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QSplitter::handle {{ background:{PAPER}; width:8px; height:8px; }}
QCheckBox {{ spacing:6px; background:transparent; }}
QStatusBar {{ background:{PAPER}; color:{MUTED}; font-size:9pt; }}
QToolTip {{ background:white; color:{INK}; border:1px solid {EDGE}; padding:5px; }}
"""
