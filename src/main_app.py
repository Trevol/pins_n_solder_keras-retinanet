import sys

from widgets.TechProcessTrackingApplication import TechProcessTrackingApplication


def main():
    app = TechProcessTrackingApplication()
    sys.exit(app.exec())


main()
