import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


def car_gui_demo():
    # Create drop down menu for microphone
    mic_label = QLabel('Select Microphone:')
    mic_menu = QComboBox()
    available_microphones = ["Microphone 1", "Microphone 2", "Microphone 3"]
    # Add in items into the box
    mic_menu.addItems(available_microphones)

    # Set up widgets
    settings_button = QPushButton('Settings')
    input_label = QLabel('Input:')
    waveform_image = QLabel()
    waveform_image.setPixmap(QPixmap("photos/spectrogram.png"))
    play_button = QPushButton('Play')
    stop_button = QPushButton('Stop')
    title_label = QLabel("Engine Recommendations")

    # left layout includes the microphone drop down menu, spectrogram image, play, stop, buttons
    left_layout = QVBoxLayout()

    # Setup mic drop down button
    mic_button = QVBoxLayout()
    mic_button.addWidget(mic_label)
    mic_button.addWidget(mic_menu)

    # horizontal layout for microphone and setting button
    top_buttons = QHBoxLayout()
    top_buttons.addLayout(mic_button)
    top_buttons.addWidget(settings_button)
    left_layout.addLayout(top_buttons)

    # add input and spectrogram images to left side
    left_layout.addWidget(input_label)
    left_layout.addWidget(waveform_image)

    # add play and stop button
    button_layout = QHBoxLayout()
    button_layout.addWidget(play_button)
    button_layout.addWidget(stop_button)
    left_layout.addLayout(button_layout)

    # Align the left side to center
    left_layout.setAlignment(Qt.AlignCenter)
    left_layout.setContentsMargins(0, 0, 0, 0)  # No margins

    # Set up the right side
    vertical_stack_layout = QVBoxLayout()
    vertical_stack_layout.setAlignment(Qt.AlignTop)  # Align to the top
    vertical_stack_layout.setContentsMargins(0, 0, 0, 0)  # No margins

    # Add items to vertical stack layout
    # Increase font side of title
    title_font = QFont()
    title_font.setPointSize(14)
    title_label.setFont(title_font)
    title_label.setAlignment(Qt.AlignCenter)

    # Add in images of car engines
    vertical_stack_layout.addWidget(title_label)
    vertical_stack_layout.addWidget(create_image_box("photos/carA.jpg", "2007 Lexus GS350 AWD | 3.0L V6"))
    vertical_stack_layout.addWidget(create_image_box("photos/carB.jpg", "2008 Scion Tc | 2.4L I4"))
    vertical_stack_layout.addWidget(create_image_box("photos/carC.jpg", "2018 Toyota Highlander | 3.0L V6"))
    vertical_stack_layout.setAlignment(Qt.AlignCenter)

    main_window = QWidget()
    # Create main layout
    main_layout = QHBoxLayout()
    main_layout.addLayout(left_layout)
    main_layout.addLayout(vertical_stack_layout)
    main_window.setLayout(main_layout)
    main_window.setWindowTitle('Car GUI Demo')
    main_window.setGeometry(100, 100, 1000, 600)

    return main_window


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
    window = car_gui_demo()
    window.show()
    app.exec_()
