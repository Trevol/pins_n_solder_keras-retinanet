from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QFormLayout, QLabel, QSizePolicy, QGroupBox, QLineEdit, QTableWidget, \
    QAbstractItemView, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QPushButton

from techprocess_tracking.TechProcesLogRecord import TechProcesLogRecord


class TechProcessInfoWidget(QGroupBox):
    def __init__(self):
        super(TechProcessInfoWidget, self).__init__()

        self.framePos = -1
        self.framePosMsec = 0.0

        self.setTitle('Log')
        self.setFixedWidth(420)

        vboxLayout = QVBoxLayout()
        self.setLayout(vboxLayout)

        buttonsLayout = QHBoxLayout()
        vboxLayout.addLayout(buttonsLayout)
        buttonsLayout.addWidget(QPushButton('Start'))
        buttonsLayout.addWidget(QPushButton('Stop'))
        buttonsLayout.addStretch(1)

        formLayout = QFormLayout()
        vboxLayout.addLayout(formLayout)

        self.framePosWidget = QLineEdit()
        self.framePosWidget.setReadOnly(True)
        formLayout.addRow('Frame:', self.framePosWidget)

        self.frameMsecWidget = QLineEdit()
        self.frameMsecWidget.setReadOnly(True)
        formLayout.addRow('MSec:', self.frameMsecWidget)

        self.pinsCountWidget = QLineEdit()
        self.pinsCountWidget.setReadOnly(True)
        formLayout.addRow('Pins:', self.pinsCountWidget)

        self.pinsWithSolderCountWidget = QLineEdit()
        self.pinsWithSolderCountWidget.setReadOnly(True)
        formLayout.addRow('With solder:', self.pinsWithSolderCountWidget)

        self.logsWidget = QTableWidget()
        vboxLayout.addWidget(self.logsWidget)
        self.logsWidget.verticalHeader().setVisible(False)
        self.logsWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.logsWidget.setColumnCount(5)
        self.logsWidget.setHorizontalHeaderLabels(
            ['MSec', 'Num of pins', 'Pins Added', 'Num of pins with solder', 'Solder added'])

    def setInfo(self, pos, msec, pinsCount, pinsWithSolderCount, logRecord: TechProcesLogRecord):
        self.framePosWidget.setText(str(pos))
        self.frameMsecWidget.setText(f'{msec:.2f}')
        self.pinsCountWidget.setText(str(pinsCount))
        self.pinsWithSolderCountWidget.setText(str(pinsWithSolderCount))

        if logRecord:
            rowCount = self.logsWidget.rowCount()
            self.logsWidget.setRowCount(rowCount + 1)
            lastRow = rowCount
            self.logsWidget.setItem(lastRow, 0, QTableWidgetItem(f'{logRecord.framePosMs:.2f}'))
            self.logsWidget.setItem(lastRow, 1, QTableWidgetItem(f'{logRecord.pinsCount}'))
            self.logsWidget.setItem(lastRow, 2, QTableWidgetItem(f'{logRecord.pinsAdded}'))
            self.logsWidget.setItem(lastRow, 3, QTableWidgetItem(f'{logRecord.pinsWithSolderCount}'))
            self.logsWidget.setItem(lastRow, 4, QTableWidgetItem(f'{logRecord.solderAdded}'))
