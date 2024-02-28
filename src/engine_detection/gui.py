import os
import shutil
import sys
import time

import scipy
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from engine_detection.engine_classifier import EngineClassifier
from engine_detection.engine_detector import EngineDetector
from engine_detection.mfcc_extractor import MFCCExtractor
from engine_detection.sound_recorder import SoundRecorder

RECORDING_LENGTH = 1  # in seconds

class GUI(QMainWindow):

    def __init__(self):
        super().__init__()

        # Create drop down menu for microphone
        mic_label = QLabel('Select Microphone:')
        self.mic_menu = QComboBox()
        available_microphones = SoundRecorder.get_devices()
        # Add in items into the box
        self.mic_menu.addItems(available_microphones)
        # mic_menu.setFixedWidth(200)

        # Set up widgets
        #settings_button = QPushButton('Settings')
        #input_label = QLabel('Spectrogram View:')
        #waveform_image = QLabel()
        #waveform_image.setPixmap(QPixmap("photos/spectrogram.png"))
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.start_button.released.connect(self.__handle_start)
        self.stop_button.released.connect(self.__handle_stop)
        title_label = QLabel("Most Likely Brands")

        # left layout includes the microphone drop down menu, spectrogram image, play, stop, buttons
        left_layout = QVBoxLayout()

        # Setup mic drop down button
        mic_button = QVBoxLayout()
        mic_button.addWidget(mic_label)
        mic_button.addWidget(self.mic_menu)

        # horizontal layout for microphone and setting button
        top_buttons = QHBoxLayout()
        top_buttons.addLayout(mic_button)
        #top_buttons.addWidget(settings_button)
        left_layout.addLayout(top_buttons)

        # add input and spectrogram images to left side
        #left_layout.addWidget(input_label)
        #left_layout.addWidget(waveform_image)

        # add play and stop button
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
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
        vertical_stack_layout.addWidget(self.__setup_image_box_1())
        vertical_stack_layout.addWidget(self.__setup_image_box_2())
        vertical_stack_layout.addWidget(self.__setup_image_box_3())
        #vertical_stack_layout.addWidget(create_image_box("photos/carA.png", "2007 Lexus GS350 AWD | 3.0L V6"))
        #vertical_stack_layout.addWidget(create_image_box("photos/carB.png", "2008 Scion Tc | 2.4L I4"))
        #vertical_stack_layout.addWidget(create_image_box("photos/carC.jpeg", "2018 Toyota Highlander | 3.0L V6"))
        vertical_stack_layout.setAlignment(Qt.AlignCenter)

        main_window = QWidget()
        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(vertical_stack_layout)
        main_window.setLayout(main_layout)
        main_window.setWindowTitle('Car GUI Demo')
        main_window.setGeometry(100, 100, 1000, 600)

        self.setCentralWidget(main_window)

        # Non-GUI setup
        self.continue_flag = False
        self.microphone = SoundRecorder()
        self.detector = EngineDetector()
        self.classifier = EngineClassifier()
        self.mfcc_extractor = MFCCExtractor()
        self.brands = ['Lexus', 'Nissan', 'Scion', 'Toyota']
        self.brands_image_dict = {
            'Lexus': 'photos/lexus.png',
            'Nissan': 'photos/nissan.png',
            'Scion': 'photos/scion.png',
            'Toyota': 'photos/toyota.png',
        }

        # Set up the temp directory
        if not os.path.exists(r'eng_temp/'):
            os.makedirs(r'eng_temp/')
        else:
            shutil.rmtree(r'eng_temp/cut_audio')
            os.makedirs(r'eng_temp/cut_audio')

    def __process_loop(self):
        while self.continue_flag:
            print("Processing")
            # Record 5 seconds of audio, then save it
            #recording = self.microphone.record(RECORDING_LENGTH)
            #SoundRecorder.preprocess(recording, 1, 48000)


            # Our audio file is saved as output.wav in the current working directory
            # Feed the audio into the engine detector. If nothing is detected, don't
            # feed it into the neural network
            if not self.detector.detect():
                continue

            # Extract the MFCCs from the file that was saved,
            # then feed them into the engine classifier
            self.mfcc_extractor.slice_audio()
            mfccs = self.mfcc_extractor.extract()
            classification_results = self.classifier.classify(mfccs)

            results_accumulator = [0] * len(classification_results[0])

            # add the results for each class
            for result in classification_results:
                for index, value in enumerate(result):
                    results_accumulator[index] += value

            #sorted_results = sorted(results_accumulator)
            sorted_results = sorted(range(len(results_accumulator)), key=lambda k: results_accumulator[k])
            brand_1 = self.brands[sorted_results[-1]]
            brand_2 = self.brands[sorted_results[-2]]
            brand_3 = self.brands[sorted_results[-3]]

            # Update the GUI images
            self.__update_image_1(brand_1)
            self.__update_image_2(brand_2)
            self.__update_image_3(brand_3)

            QApplication.processEvents()


    def __handle_start(self):
        print('Start')
        self.continue_flag = True
        self.microphone.set_device(self.mic_menu.currentIndex())
        self.__process_loop()

    def __handle_stop(self):
        self.continue_flag = False


    def __setup_image_box_1(self):
        # Create a widget to contain the layout
        image_box_1 = QWidget()
        image_box_layout = QVBoxLayout(image_box_1)
        self.image_1_label = QLabel()
        self.image_1_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(300))
        self.image_1_description = QLabel("Brand 1: None")

        # Set alignment for the labels
        self.image_1_label.setAlignment(Qt.AlignCenter)
        self.image_1_description.setAlignment(Qt.AlignCenter)

        # Add widgets to image box layout
        image_box_layout.addWidget(self.image_1_label)
        image_box_layout.addWidget(self.image_1_description)

        return image_box_1  # Return the widget that contains the layout

    def __setup_image_box_2(self):
        # Create a widget to contain the layout
        image_box_2 = QWidget()
        image_box_layout = QVBoxLayout(image_box_2)
        self.image_2_label = QLabel()
        self.image_2_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(300))
        self.image_2_description = QLabel("Brand 2: None")

        # Set alignment for the labels
        self.image_2_label.setAlignment(Qt.AlignCenter)
        self.image_2_description.setAlignment(Qt.AlignCenter)

        # Add widgets to image box layout
        image_box_layout.addWidget(self.image_2_label)
        image_box_layout.addWidget(self.image_2_description)

        return image_box_2  # Return the widget that contains the layout

    def __setup_image_box_3(self):
        # Create a widget to contain the layout
        image_box_3 = QWidget()
        image_box_layout = QVBoxLayout(image_box_3)
        self.image_3_label = QLabel()
        self.image_3_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(300))
        self.image_3_description = QLabel("Brand 3: None")

        # Set alignment for the labels
        self.image_3_label.setAlignment(Qt.AlignCenter)
        self.image_3_description.setAlignment(Qt.AlignCenter)

        # Add widgets to image box layout
        image_box_layout.addWidget(self.image_3_label)
        image_box_layout.addWidget(self.image_3_description)

        return image_box_3  # Return the widget that contains the layout

    def __update_image_1(self, brand: str):
        self.image_1_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(300))
        self.image_1_description.setText(f'Brand 1: {brand}')

    def __update_image_2(self, brand: str):
        self.image_2_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(300))
        self.image_2_description.setText(f'Brand 2: {brand}')

    def __update_image_3(self, brand: str):
        self.image_3_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(300))
        self.image_3_description.setText(f'Brand 3: {brand}')
