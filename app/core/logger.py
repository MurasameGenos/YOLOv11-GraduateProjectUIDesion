import logging
from PySide6.QtCore import QObject, Signal

class LogEmitter(QObject):
    message = Signal(str)

class QtHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.message.emit(msg)
