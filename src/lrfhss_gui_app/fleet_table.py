"""Cached fleet cells and painted validity chips; no reset on refresh."""
from contextlib import contextmanager
from PySide6 import QtCore,QtGui,QtWidgets
from .health import SENSORS
from .theme import GOOD,MUTED,MINT,PANEL,TEXT

SORT_ROLE=QtCore.Qt.UserRole+1
STALE_ROLE=QtCore.Qt.UserRole+2
CHIPS_ROLE=QtCore.Qt.UserRole+3

class FleetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self,other):
        a,b=self.data(SORT_ROLE),other.data(SORT_ROLE)
        return a<b if a is not None and b is not None else super().__lt__(other)

class FleetDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self,painter,option,index):
        painter.save()
        if index.data(STALE_ROLE):painter.setOpacity(.45)
        if index.column()!=5:
            super().paint(painter,option,index)
        else:
            opt=QtWidgets.QStyleOptionViewItem(option);self.initStyleOption(opt,index);opt.text=''
            style=opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
            style.drawControl(QtWidgets.QStyle.CE_ItemViewItem,opt,painter,opt.widget)
            mask=index.data(CHIPS_ROLE) or 0
            size=18;left=option.rect.left()+5;top=option.rect.center().y()-9
            for i,(_,letter,_) in enumerate(SENSORS):
                valid=bool(mask&(1<<i));rect=QtCore.QRect(left+i*20,top,size,18)
                painter.fillRect(rect,QtGui.QColor(GOOD if valid else '#e4eaeb'))
                painter.setPen(QtGui.QColor(PANEL if valid else MUTED))
                painter.drawText(rect,QtCore.Qt.AlignCenter,letter)
        painter.restore()

class FleetTable(QtWidgets.QTableWidget):
    def __init__(self,parent=None):
        super().__init__(0,7,parent);self.items_by_id={};self.cache={}
        self.cell_updates=0;self.change_signals=0
        self._batch_depth=0;self._sort_paused=False
        self.setItemDelegate(FleetDelegate(self))

    @contextmanager
    def batch(self):
        if self._batch_depth:
            yield
            return
        self._batch_depth=1;self._sort_paused=False
        self._sorting=self.isSortingEnabled();self._sort_column=self.horizontalHeader().sortIndicatorSection()
        v=self.verticalScrollBar().value();h=self.horizontalScrollBar().value()
        try:yield
        finally:
            if self._sort_paused:self.setSortingEnabled(True)
            self._batch_depth=0
            self.verticalScrollBar().setValue(v);self.horizontalScrollBar().setValue(h)

    def update_cells(self,csid,changes):
        if not self._batch_depth:
            with self.batch():return self.update_cells(csid,changes)
        cached=self.cache.setdefault(csid,{})
        changed={col:value for col,value in changes.items() if cached.get(col)!=value}
        if not changed:return 0
        item=self.items_by_id.get(csid)
        # Suspending/restarting sorting on every frame emits layoutChanged and
        # repaints the entire viewport. Pause only for insertion or a changed
        # sort key; ordinary age/counter/validity updates retain the layout.
        key=self._sort_column
        key_changed=key in changed and (key not in cached or cached[key][1]!=changed[key][1])
        if self._sorting and not self._sort_paused and (item is None or key_changed):
            self.setSortingEnabled(False);self._sort_paused=True
        if item is None:
            row=self.rowCount();self.insertRow(row)
            for c in range(7):self.setItem(row,c,FleetItem())
            item=self.item(row,0);item.setData(QtCore.Qt.UserRole,csid);self.items_by_id[csid]=item
        row=self.row(item);model=self.model();blocked=model.blockSignals(True)
        try:
            for col,value in changed.items():
                text,sort,stale,mask=value;cell=self.item(row,col)
                cell.setData(QtCore.Qt.DisplayRole,text);cell.setData(SORT_ROLE,sort)
                cell.setData(STALE_ROLE,stale)
                if col==5:
                    cell.setData(CHIPS_ROLE,mask)
                    cell.setToolTip(' · '.join(f'{name}: {"valid" if mask&(1<<i) else "invalid or missing"}' for i,(_,_,name) in enumerate(SENSORS)))
                cached[col]=value
        finally:model.blockSignals(blocked)
        self.cell_updates+=len(changed);self.change_signals+=1
        model.dataChanged.emit(model.index(row,min(changed)),model.index(row,max(changed)),[])
        return len(changed)

    def upsert(self,csid,values,sorts,mask,stale):
        self.update_cells(csid,{c:(text,sorts[c],stale,mask if c==5 else 0) for c,text in enumerate(values)})
        return self.items_by_id[csid]

    def age(self,csid,label,seconds,stale):
        cached=self.cache.get(csid)
        if not cached:return
        changes={3:(label,int(seconds),stale,0)}
        if cached[0][2]!=stale:
            changes.update({c:(v[0],v[1],stale,v[3]) for c,v in cached.items() if c!=3})
        self.update_cells(csid,changes)

    def clear_rows(self):
        self.setRowCount(0);self.items_by_id.clear();self.cache.clear()
