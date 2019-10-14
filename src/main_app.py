import sys

from models.ModelsContext import ModelsContext
from widgets.TechProcessTrackingApplication import TechProcessTrackingApplication


def main():
    rootCtx = ModelsContext()
    app = TechProcessTrackingApplication()
    sys.exit(app.exec())


main()
