"""Preserve an application quit request across asynchronous window cleanup."""
from PySide6 import QtCore


class QuitRelay(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.application = QtCore.QCoreApplication.instance()
        self.requested = False
        self.application.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.application and event.type() == QtCore.QEvent.Type.Quit:
            self.requested = True
        return False

    def window_closed(self):
        # QApplication.quit() first closes top-level windows. A close rejected
        # while flushing asynchronous output consumes that quit request. Retry
        # it after cleanup even when quitOnLastWindowClosed is disabled.
        if self.requested:
            self.requested = False
            QtCore.QTimer.singleShot(0, self.application.quit)
