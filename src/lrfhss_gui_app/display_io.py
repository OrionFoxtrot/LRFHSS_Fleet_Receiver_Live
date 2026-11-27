"""GUI-side disk work. Decoder ownership and journal formats remain unchanged."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from PySide6 import QtCore
from .journal import Delivery


class FileJobs(QtCore.QObject):
    """Serialize settings/session/export I/O outside the Qt event loop."""
    completed=QtCore.Signal(int,object,object)
    idle=QtCore.Signal()
    quiescent=QtCore.Signal()
    notice=QtCore.Signal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.executor=None
        self.pending={};self.sequence=0
        self.details={};self._workers=set();self._retiring=set()
        self._lock=threading.Lock();self._closed=False;self._stopping=False
        self._deadline=None;self._cancel_notified=False
        self.join_timer=QtCore.QTimer(self);self.join_timer.setInterval(20)
        self.join_timer.timeout.connect(self._poll_shutdown)
        self.completed.connect(self._complete,QtCore.Qt.ConnectionType.QueuedConnection)

    def submit(self,work,done,*,label=None,cancel=None):
        if self._closed:raise RuntimeError('GUI file jobs are closed')
        if self.executor is None:
            self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="GUI file I/O")
        self.sequence+=1;key=self.sequence;self.pending[key]=done
        detail=dict(label=label or getattr(work,'__name__','GUI file job'),
                    submitted=time.monotonic(),started=None,cancel=cancel,future=None)
        self.details[key]=detail
        def run():
            with self._lock:self._workers.add(threading.current_thread())
            detail['started']=time.monotonic()
            return work()
        try:future=self.executor.submit(run)
        except Exception:
            self.pending.pop(key);self.details.pop(key)
            raise
        detail['future']=future
        def finished(value):
            try:result,error=value.result(),None
            except CancelledError:result,error=None,'Cancelled GUI output during shutdown; no partial CSV was published.'
            except Exception as exc:result,error=None,str(exc)
            self.completed.emit(key,result,error)
        future.add_done_callback(finished)
        return key

    @QtCore.Slot(int,object,object)
    def _complete(self,key,result,error):
        callback=self.pending.pop(key)
        self.details.pop(key)
        try:callback(result,error)
        finally:
            if not self.pending:self.idle.emit()
            if self._stopping:self._poll_shutdown()

    def snapshot(self):
        now=time.monotonic()
        return [dict(key=key,label=d['label'],age_s=now-d['submitted'],
                     running=d['started'] is not None,
                     cancellation_requested=bool(d['cancel'] and d['cancel'].is_set()))
                for key,d in self.details.items()]

    @property
    def is_quiescent(self):
        with self._lock:threads=self._workers | self._retiring
        return not self.pending and self.executor is None and not any(t.is_alive() for t in threads)

    def quiesce(self,*,cancel_after_s=5.):
        """Drain output, then join the idle worker without blocking Qt.

        Exports cooperate with cancellation after the grace interval. Critical
        summary/settings writes finish atomically; they are never discarded.
        Each thread join is a zero-time poll, so the GUI remains responsive.
        """
        if self.is_quiescent:return
        if not self._stopping:
            self._stopping=True;self._deadline=time.monotonic()+cancel_after_s
            self._cancel_notified=False
        self.join_timer.start();self._poll_shutdown()

    @QtCore.Slot()
    def _poll_shutdown(self):
        if not self._stopping:return
        if self.pending:
            if time.monotonic()>=self._deadline:
                cancelled=[]
                for d in list(self.details.values()):
                    if d['cancel'] is not None:
                        d['cancel'].set();cancelled.append(d['label'])
                        if d['future'] is not None:d['future'].cancel()
                if not self._cancel_notified:
                    self.notice.emit('Cancelling pending export; completing session metadata.' if cancelled
                                     else 'Waiting for session metadata to finish writing.')
                    self._cancel_notified=True
            return
        if self.executor is not None:
            self.executor.shutdown(wait=False);self.executor=None
            with self._lock:
                self._retiring.update(self._workers);self._workers.clear()
        with self._lock:threads=list(self._retiring)
        for thread in threads:thread.join(timeout=0)
        with self._lock:self._retiring={t for t in self._retiring if t.is_alive()}
        if self.is_quiescent:
            self._stopping=False;self.join_timer.stop();self.quiescent.emit()

    def close(self):
        self.quiesce()
        # A preparation callback may still need to queue the final summary.
        # Seal the queue only after callbacks and the worker have both drained.
        if self.is_quiescent:self._closed=True
        return self._closed


@dataclass
class Receipt:
    accepted: threading.Event=field(default_factory=threading.Event)
    error: str | None=None


class JournalReader(QtCore.QThread):
    """Read at most one unacknowledged batch; Qt owns the model mutation.

    Delivery advances its packet cursor only after the GUI accepts the batch.
    All tail reads, leases, final checks and completion writes run here.
    """
    packet_batch=QtCore.Signal(object,object)
    event_batch=QtCore.Signal(object,object)
    fault=QtCore.Signal(str)
    opened=QtCore.Signal()

    def __init__(self,journal,*,complete_probe=None,parent=None):
        super().__init__(parent)
        self.journal=journal;self.complete_probe=complete_probe
        self.owner_done=threading.Event();self.cancel=threading.Event()
        self.normal_completion=False
        self.owner_exitcode=None
        self.pending_delivery=None

    def _deliver(self,signal,values):
        receipt=Receipt()
        self.pending_delivery=dict(started=time.monotonic(),items=len(values))
        try:
            signal.emit(values,receipt)
            while not receipt.accepted.wait(.05):
                if self.cancel.is_set():raise InterruptedError("Display delivery stopped")
            if receipt.error:raise RuntimeError(receipt.error)
        finally:self.pending_delivery=None

    def snapshot(self):
        pending=self.pending_delivery
        return dict(running=self.isRunning(),finished=self.isFinished(),
                    pending_items=pending['items'] if pending else 0,
                    pending_age_s=time.monotonic()-pending['started'] if pending else 0,
                    owner_done=self.owner_done.is_set(),cancel=self.cancel.is_set())

    def run(self):
        reader=None
        try:
            try:
                reader=Delivery(self.journal)
            except (PermissionError,BlockingIOError):
                if self.complete_probe is not None:
                    return  # Another live window owns this recovery session.
                raise
            self.opened.emit()
            while not self.cancel.is_set():
                events=[]
                more=reader.drain(lambda rows:self._deliver(self.packet_batch,rows),lambda kind,value:events.append((kind,value)))
                if events:self._deliver(self.event_batch,events)
                done=self.complete_probe() if self.complete_probe is not None else self.owner_done.is_set()
                if done and not more:
                    # The owner can commit between the first drain and exit.
                    events=[]
                    more=reader.drain(lambda rows:self._deliver(self.packet_batch,rows),lambda kind,value:events.append((kind,value)))
                    if events:self._deliver(self.event_batch,events)
                    if not more:
                        if self.complete_probe is None and not (self.journal.directory/'finished.json').exists():
                            self._deliver(self.event_batch,[('error',
                                f'Receiver owner exited unexpectedly ({self.owner_exitcode}).')])
                        reader.finish();reader=None;self.normal_completion=True;break
                if not more:self.cancel.wait(.25)
        except InterruptedError:
            pass  # An unfinished journal is deliberately recoverable.
        except Exception as exc:
            self.fault.emit(f"Receiver display stopped. The journal is available for recovery. {exc}")
        finally:
            if reader is not None:reader.close()
