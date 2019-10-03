from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QWidget, QFormLayout, QLineEdit, QTableWidget, QVBoxLayout, QButtonGroup, \
    QHBoxLayout, QPushButton, QTableWidgetItem, QAbstractItemView


class Window(QWidget):
    def keyPressEvent(self, keyEvent: QKeyEvent):
        if keyEvent.key() == Qt.Key_Escape:
            self.close()

    def __init__(self):
        super(Window, self).__init__()
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

        logsWidget = QTableWidget()
        vboxLayout.addWidget(logsWidget)
        logsWidget.verticalHeader().setVisible(False)
        logsWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        logsWidget.setColumnCount(3)
        logsWidget.setHorizontalHeaderLabels(['11', '22' , '33'])
        logsWidget.setRowCount(1)
        logsWidget.setItem(0, 0, QTableWidgetItem('0_0'))
        logsWidget.setItem(0, 1, QTableWidgetItem('0_1'))
        logsWidget.setItem(0, 2, QTableWidgetItem('0_2'))

def main():
    app = QApplication([])
    w = Window()
    w.show()
    app.exec()


main()
