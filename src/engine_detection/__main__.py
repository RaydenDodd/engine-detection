import sys
from PyQt5.QtWidgets import QApplication

from engine_detection.gui import GUI

if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    sys.exit(app.exec())
