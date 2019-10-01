import sys
from PyQt5.QtWidgets import QApplication

from widgets import MainWindow


class TechProcessTrackingApplication(QApplication):
    def __init__(self):
        super(TechProcessTrackingApplication, self).__init__([])
        self.mainWindow = MainWindow()


    def exec(self):
        self.mainWindow.showMaximized()
        return super(TechProcessTrackingApplication, self).exec()
