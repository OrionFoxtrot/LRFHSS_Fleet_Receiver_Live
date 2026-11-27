"""Compact Qt operator widgets; setters avoid unchanged widget updates."""
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .theme import BAD, BORDER, GOOD, MINT, MUTED, PANEL, TEXT, operator_font


def set_text(label, text):
    if label.text() != text: label.setText(text)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title, value="--", sub="", parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setMinimumHeight(59)
        lay=QtWidgets.QVBoxLayout(self);lay.setContentsMargins(9,5,9,5);lay.setSpacing(0)
        self.title=QtWidgets.QLabel(title)
        self.title.setObjectName("overline")
        self.value=QtWidgets.QLabel(value);self.value.setFont(operator_font(16,mono=True,bold=True))
        self.sub=QtWidgets.QLabel(sub);self.sub.setStyleSheet(f"color:{MUTED};font-size:8pt;")
        for label in (self.title,self.value,self.sub):lay.addWidget(label)
        self.set_value(value,sub)

    def set_value(self,value,sub=None,color=None):
        set_text(self.value,value)
        style=f"color:{color or TEXT};font-family:Consolas,'Courier New';font-size:16pt;font-weight:700;"
        if self.value.styleSheet()!=style:self.value.setStyleSheet(style)
        if sub is not None:set_text(self.sub,sub)


class BannerMetric(QtWidgets.QWidget):
    def __init__(self,title,parent=None):
        super().__init__(parent)
        lay=QtWidgets.QHBoxLayout(self);lay.setContentsMargins(0,0,0,0);lay.setSpacing(7)
        self.value=QtWidgets.QLabel("0");self.value.setFont(operator_font(21,mono=True,bold=True))
        self.title=QtWidgets.QLabel(title);self.title.setFont(operator_font(10))
        self.sub=QtWidgets.QLabel(self);self.sub.hide()
        lay.addWidget(self.value);lay.addWidget(self.title)
        self.set_value('0')

    def set_value(self,value,sub=None,color=None):
        set_text(self.value,value)
        style=f"color:{'#ffb6a6' if color==BAD else '#f6fbfa'};font-family:Consolas,'Courier New';font-size:21pt;font-weight:700;"
        if self.value.styleSheet()!=style:self.value.setStyleSheet(style)
        if sub is not None:
            set_text(self.sub,sub)
            if self.toolTip()!=sub:self.setToolTip(sub)


class HealthCell(QtWidgets.QWidget):
    def __init__(self,title,value="—",sub="",parent=None):
        super().__init__(parent)
        lay=QtWidgets.QVBoxLayout(self);lay.setContentsMargins(0,0,0,0);lay.setSpacing(2)
        self.title=QtWidgets.QLabel(title);self.title.setStyleSheet(f"color:{MUTED};font-size:8pt;")
        self.value=QtWidgets.QLabel(value);self.value.setFont(operator_font(9,mono=True))
        self.sub=QtWidgets.QLabel(sub,self);self.sub.hide()
        lay.addWidget(self.title);lay.addWidget(self.value)
        self.set_value(value,sub)

    def set_value(self,value,sub=None,color=None):
        set_text(self.value,value)
        style=f"color:{color or TEXT};font-family:Consolas,'Courier New';font-size:9pt;"
        if self.value.styleSheet()!=style:self.value.setStyleSheet(style)
        if sub is not None:
            set_text(self.sub,sub)
            if self.toolTip()!=sub:self.setToolTip(sub)


class ValidityPill(QtWidgets.QLabel):
    def __init__(self,name):
        super().__init__(name);self.name=name
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(22);self.set_state(None)

    def set_state(self,state):
        if hasattr(self,"_state") and self._state is state:return
        self._state=state
        fg,bg=(GOOD,MINT) if state is True else (BAD,"#fbe8e3") if state is False else (MUTED,"#f4f7f7")
        self.setStyleSheet(f"color:{fg};background:{bg};border:1px solid {BORDER};"
                          "padding:2px 5px;font-size:8pt;font-weight:600;")


class PlotPane(pg.PlotWidget):
    def __init__(self,title,y_label="",parent=None):
        super().__init__(parent=parent)
        self.setBackground(PANEL);self.showGrid(x=True,y=True,alpha=.18)
        self.setTitle(title,color=TEXT,size="10pt")
        if y_label:self.setLabel("left",y_label,color=MUTED)
        self.setLabel("bottom","Samples",color=MUTED)
        for name in ("left","bottom"):
            self.getAxis(name).setPen(BORDER);self.getAxis(name).setTextPen(MUTED)
            self.getAxis(name).setStyle(tickFont=operator_font(8,mono=True))
        self.setMinimumHeight(110)
