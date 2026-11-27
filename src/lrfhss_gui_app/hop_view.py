"""Bounded O(1) packet accounting; widgets render only on demand."""
from collections import Counter
import time
from PySide6 import QtCore,QtGui,QtWidgets
from .theme import BAD,TEXT

BIN_SECONDS=60
BIN_COUNT=30

class HopBook:
    def __init__(self,ids=(256,)):
        self.ids=tuple(ids);self.bins={};self.totals=Counter();self.nodes={};self.version=0
    def configure(self,ids):
        if tuple(ids)!=self.ids:self.ids=tuple(ids);self.version+=1
    def add(self,envelope):
        hop=envelope.get('phy',{}).get('hop_seq_id')
        if not isinstance(hop,int) or not 0<=hop<=511:return
        minute=int(envelope.get('received_epoch',time.time())//BIN_SECONDS)
        slots=self.bins.get(hop)
        if slots is None:
            slots=self.bins[hop]=[[None,0] for _ in range(BIN_COUNT)]
        slot=slots[minute%BIN_COUNT]
        if slot[0]!=minute:slot[:]=[minute,0]
        slot[1]+=1;self.totals[hop]+=1;self.version+=1
        record=envelope.get('record',{});d=record.get('decoded',{})
        if not envelope.get('phy',{}).get('crc_ok') or record.get('packet_type')!='DeSCENT_V2_5' or 'csid' not in d:return
        csid=int(d['csid']);row=self.nodes.setdefault(csid,dict(hop=hop,expected=None,mismatches=0))
        expected=self.ids[csid%2] if len(self.ids)==2 else None
        row.update(hop=hop,expected=expected)
        if expected is not None and hop!=expected:row['mismatches']+=1
    def count(self,hop,minute):
        slots=self.bins.get(hop)
        if slots is None:return 0
        tag,count=slots[minute%BIN_COUNT]
        return count if tag==minute else 0

class HopView(QtWidgets.QWidget):
    def __init__(self,book,parent=None):
        super().__init__(parent);self.book=book;self.draw_count=0;self._drawn=None;self._cells={}
        layout=QtWidgets.QVBoxLayout(self)
        controls=QtWidgets.QHBoxLayout();self.live_update=QtWidgets.QCheckBox('Live update');self.live_update.setChecked(False)
        self.refresh=QtWidgets.QPushButton('Refresh');self.refresh.clicked.connect(lambda:self.render(force=True))
        controls.addWidget(self.live_update);controls.addWidget(self.refresh)
        self.note=QtWidgets.QLabel('One-minute bins · newest first · last 30 minutes. Totals count all session packets with a hop ID.')
        controls.addWidget(self.note,1);layout.addLayout(controls)
        self.counts=QtWidgets.QTableWidget(0,BIN_COUNT+2);self.counts.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.counts.verticalHeader().hide();layout.addWidget(self.counts)
        layout.addWidget(QtWidgets.QLabel('Per ChipSat · trusted payloads · even CSID → first ID, odd → second ID'))
        self.nodes=QtWidgets.QTableWidget(0,4);self.nodes.setHorizontalHeaderLabels(['CSID','Last hop ID','Parity check','Mismatches'])
        self.nodes.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers);self.nodes.verticalHeader().hide()
        self.nodes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch);layout.addWidget(self.nodes,1)
        self.live_update.toggled.connect(lambda checked:self.maybe_refresh())
    def maybe_refresh(self):
        if self.isVisible() and self.live_update.isChecked():self.render()
    def render(self,force=False,now=None):
        if not self.isVisible() or (not force and not self.live_update.isChecked()):return False
        now=time.time() if now is None else now;minute=int(now//BIN_SECONDS)
        signature=(self.book.version,minute)
        if not force and self._drawn==signature:return False
        self._drawn=signature;self.draw_count+=1
        minutes=range(minute,minute-BIN_COUNT,-1)
        self.counts.setHorizontalHeaderLabels(['Hop ID','Total']+[time.strftime('%H:%M',time.localtime(m*BIN_SECONDS)) for m in minutes])
        self.counts.setRowCount(len(self.book.ids))
        for row,hop in enumerate(self.book.ids):
            for col,v in enumerate([hop,self.book.totals[hop]]+[self.book.count(hop,m) for m in minutes]):self._set(self.counts,row,col,str(v))
        self.counts.setColumnWidth(0,65);self.counts.setColumnWidth(1,70)
        for col in range(2,BIN_COUNT+2):self.counts.setColumnWidth(col,57)
        self.nodes.setRowCount(len(self.book.nodes))
        for row,(csid,data) in enumerate(sorted(self.book.nodes.items())):
            expected=self.book.ids[csid%2] if len(self.book.ids)==2 else None
            mismatch=expected is not None and data['hop']!=expected
            parity='n/a · Single' if len(self.book.ids)==1 else f'{"MISMATCH" if mismatch else "OK"} · expected {expected}' if expected is not None else 'n/a'
            for col,v in enumerate((csid,data['hop'],parity,data['mismatches'])):self._set(self.nodes,row,col,str(v),BAD if data['mismatches'] else TEXT)
        return True
    def _set(self,table,row,col,text,color=TEXT):
        key=(id(table),row,col);value=(text,color)
        if self._cells.get(key)==value and table.item(row,col) is not None:return
        item=table.item(row,col)
        if item is None:item=QtWidgets.QTableWidgetItem();table.setItem(row,col,item)
        item.setText(text);item.setForeground(QtGui.QColor(color));self._cells[key]=value
