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

        self.pinsCountWidget = QLineEdit()
        self.pinsCountWidget.setReadOnly(True)
        layout.addRow('Pins:', self.pinsCountWidget)

        self.pinsWithSolderCountWidget = QLineEdit()
        self.pinsWithSolderCountWidget.setReadOnly(True)
        layout.addRow('With solder:', self.pinsWithSolderCountWidget)

    def setInfo(self, pos, msec, pinsCount, pinsWithSolderCount, logRecord):
        self.framePosWidget.setText(str(pos))
        self.frameMsecWidget.setText(f'{msec:.2f}')
        self.pinsCountWidget.setText(str(pinsCount))
        self.pinsWithSolderCountWidget.setText(str(pinsWithSolderCount))
