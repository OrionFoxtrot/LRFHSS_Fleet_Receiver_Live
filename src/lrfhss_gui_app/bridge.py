"""Keep the decoder owner in a different interpreter from the visible Qt window."""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from PySide6 import QtCore
from .journal import Delivery, Writer, durable_json

NOTICE_CAPACITY = 8


def receive(config,journal,hints,stop):
    from .receiver import ReceiverWorker
    parent=mp.parent_process()
    hints.cancel_join_thread()
    writer=Writer(journal,hints)
    worker=ReceiverWorker(config)
    # The packet callback propagates journal I/O errors through ReceiverWorker.run.
    publication_errors=[]
    def publish(envelope):
        try:
            writer.packet(envelope)
        except OSError as exc:
            publication_errors.append(str(exc))
            raise
    worker.publish_packet=publish
    errors=[]
    def error(value):
        errors.append(value)
        writer.event("error",value)
    worker.error.connect(error,QtCore.Qt.ConnectionType.DirectConnection)
    last_stats=None
    def status(value):
        nonlocal last_stats
        writer.event("status",value)
        last_stats=value
    worker.receiver_stats.connect(status,QtCore.Qt.ConnectionType.DirectConnection)
    for signal,name in ((worker.message,"message"),(worker.autoset_result,"autoset_result")):
        signal.connect(lambda v,kind=name:writer.event(kind,v),QtCore.Qt.ConnectionType.DirectConnection)
    exited=threading.Event()
    def watch_stop():
        while not exited.is_set():
            if stop.wait(.1) or parent is not None and not parent.is_alive():
                worker.request_stop()
                return
    watcher=threading.Thread(target=watch_stop,name="receiver-stop",daemon=True)
    watcher.start()
    try:
        worker.run()
        journal.finish_summary(last_stats)
        writer.event("finished")
        durable_json(journal.directory/"finished.json",{"errors":errors,"publication_errors":publication_errors})
    finally:
        exited.set()
        watcher.join(1)
        writer.close()
        hints.close()


class ReceiverBridge(QtCore.QObject):
    finished=QtCore.Signal()

    def __init__(self,config,journal,accept,on_event,parent=None):
        super().__init__(parent)
        from types import SimpleNamespace
        from .display_io import JournalReader
        self.delivery=SimpleNamespace(journal=journal)
        self.accept=accept;self.on_event=on_event
        ctx=mp.get_context('spawn')
        self.hints=ctx.Queue(maxsize=NOTICE_CAPACITY);self.stop_event=ctx.Event()
        self.process=ctx.Process(target=receive,args=(config,journal,self.hints,self.stop_event),name='LRFHSS receiver owner')
        self.reader=JournalReader(journal,parent=self)
        self.reader.packet_batch.connect(self._accept,QtCore.Qt.ConnectionType.QueuedConnection)
        self.reader.event_batch.connect(self._events,QtCore.Qt.ConnectionType.QueuedConnection)
        self.reader.fault.connect(self._fault,QtCore.Qt.ConnectionType.QueuedConnection)
        self.reader.finished.connect(self._reader_finished,QtCore.Qt.ConnectionType.QueuedConnection)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(250);self.timer.timeout.connect(self.poll)
        self.polling=False;self._started=False;self._reader_done=False;self._closed=False

    def start(self):
        try:self.process.start()
        except Exception:
            self.hints.close();self.process.close();self._closed=True
            raise
        self._started=True;self.reader.start();self.timer.start()

    def request_stop(self):self.stop_event.set()

    @QtCore.Slot(object,object)
    def _accept(self,batch,receipt):
        try:self.accept(batch)
        except Exception as exc:receipt.error=str(exc)
        finally:receipt.accepted.set()

    @QtCore.Slot(object,object)
    def _events(self,events,receipt):
        try:
            for kind,value in events:self.on_event(kind,value)
        except Exception as exc:receipt.error=str(exc)
        finally:receipt.accepted.set()

    @QtCore.Slot(str)
    def _fault(self,message):
        self.request_stop();self.on_event('error',message)

    @QtCore.Slot()
    def _reader_finished(self):
        self._reader_done=True;self.reap()

    @QtCore.Slot()
    def poll(self):
        if self.polling or self._closed or not self._started:return
        self.polling=True
        try:
            for _ in range(NOTICE_CAPACITY):
                try:self.hints.get_nowait()
                except queue.Empty:break
            if not self.process.is_alive():
                self.reader.owner_exitcode=self.process.exitcode
                self.reader.owner_done.set()
            self.reap()
        finally:self.polling=False

    @QtCore.Slot()
    def reap(self):
        if self._closed or not self._started or self.process.is_alive():return
        self.reader.owner_exitcode=self.process.exitcode
        self.reader.owner_done.set()
        if not self._reader_done or not self.reader.wait(0):return
        self._closed=True;self.timer.stop();self.process.join(timeout=0);self.process.close()
        # This parent only consumes notices, so it owns no queue feeder to join.
        self.hints.close();self.finished.emit()
