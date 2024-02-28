import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from engine_detection.gui import GUI


def create_image_box(image_path, description):
    # Create a widget to contain the layout
    container_widget = QWidget()
    image_box_layout = QVBoxLayout(container_widget)
    image_label = QLabel()
    image_label.setPixmap(QPixmap(image_path).scaledToWidth(300))
    description_label = QLabel(description)

    # Set alignment for the labels
    image_label.setAlignment(Qt.AlignCenter)
    description_label.setAlignment(Qt.AlignCenter)

    # Add widgets to image box layout
    image_box_layout.addWidget(image_label)
    image_box_layout.addWidget(description_label)

    return container_widget  # Return the widget that contains the layout


if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    sys.exit(app.exec())
