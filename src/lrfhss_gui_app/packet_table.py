"""Recent packet rows backed by a bounded Qt model; one insertion signal per batch."""
from __future__ import annotations
from PySide6 import QtCore, QtGui
from .theme import BAD, GOOD

COLUMNS=("Time","Type","CSID","Counter","SoC","PHY CRC","App CRC","CFO","Bytes","Raw / Note","Hop ID")


class PacketRows(QtCore.QAbstractTableModel):
    def __init__(self,capacity,parent=None):
        super().__init__(parent)
        self.capacity=capacity
        self.rows=[]

    def rowCount(self,parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self,parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self,section,orientation,role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation==QtCore.Qt.Orientation.Horizontal and role==QtCore.Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return super().headerData(section,orientation,role)

    def data(self,index,role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value=self.rows[index.row()][index.column()]
        if role==QtCore.Qt.ItemDataRole.DisplayRole:
            return value
        if role==QtCore.Qt.ItemDataRole.ForegroundRole and index.column() in (5,6) and value in ("PASS","FAIL"):
            return QtGui.QColor(GOOD if value=="PASS" else BAD)
        return None

    def insert_packets(self,rows):
        rows=list(reversed(rows[-self.capacity:]))
        if not rows:
            return
        keep=min(len(self.rows),self.capacity-len(rows))
        if keep<len(self.rows):
            self.beginRemoveRows(QtCore.QModelIndex(),keep,len(self.rows)-1)
            del self.rows[keep:]
            self.endRemoveRows()
        self.beginInsertRows(QtCore.QModelIndex(),0,len(rows)-1)
        self.rows[:0]=rows
        self.endInsertRows()

    def clear(self):
        if self.rows:
            self.beginRemoveRows(QtCore.QModelIndex(),0,len(self.rows)-1)
            self.rows.clear()
            self.endRemoveRows()

