from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QFormLayout, QLabel, QSizePolicy, QGroupBox, QLineEdit


class TechProcessInfoWidget(QGroupBox):
    def __init__(self):
        super(TechProcessInfoWidget, self).__init__()

        self.framePos = -1
        self.framePosMsec = 0.0

        self.setTitle('Log')
        self.setFixedWidth(220)

        layout = QFormLayout()
        self.setLayout(layout)

        self.framePosWidget = QLineEdit()
        self.framePosWidget.setReadOnly(True)
        layout.addRow('Frame:', self.framePosWidget)

        self.frameMsecWidget = QLineEdit()
        self.frameMsecWidget.setReadOnly(True)
        layout.addRow('MSec:', self.frameMsecWidget)

        timer = QTimer(self)
        timer.timeout.connect(self.updateFramePos)
        timer.start(200)

        self.updateFramePos()

    def updateFramePos(self):
        self.framePos += 1
        self.framePosMsec = self.framePos * .66

        self.framePosWidget.setText(str(self.framePos))
        self.frameMsecWidget.setText(f'{self.framePosMsec:.2f}')
