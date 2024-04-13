import os
import shutil
import sys
import time

import scipy
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QTabWidget, QLineEdit
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from tkinter import *
from tkinter.filedialog import askopenfilename

from engine_detection.engine_classifier import EngineClassifier
from engine_detection.engine_detector import EngineDetector
from engine_detection.mfcc_extractor import MFCCExtractor
from engine_detection.sound_recorder import SoundRecorder

RECORDING_LENGTH = 5  # in seconds

class GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(self.script_path)

        # If the user selects to read audio from a file, this variable will be populated
        self.selected_file = None


        # Set up widgets

        title_label = QLabel("Most Likely Brands")

        ########################

        # left layout includes the microphone drop down menu, spectrogram image, play, stop, buttons
        #left_layout = QVBoxLayout()
        left_layout = self.__setup_left_layout()



        # Align the left side to center
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setContentsMargins(0, 0, 0, 0)  # No margins

        #######################3

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
        vertical_stack_layout.setAlignment(Qt.AlignCenter)

        main_window = QWidget()
        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(vertical_stack_layout)
        main_window.setLayout(main_layout)
        main_window.setWindowTitle('Engine Detection')
        main_window.setGeometry(100, 100, 1000, 600)

        self.setCentralWidget(main_window)

        # Non-GUI setup
        pickle_file_path = os.path.join(self.script_dir, '..', 'trained_models', 'label_to_category_top_mapping.pickle')
        with open(pickle_file_path, 'rb') as file:
            self.brands_mapping = pickle.load(file)
        self.continue_flag = False
        self.microphone = SoundRecorder()
        self.detector = EngineDetector()
        self.classifier = EngineClassifier()
        self.mfcc_extractor = MFCCExtractor()
        self.brands_image_dict = {brand_name: f'photos/{brand_name.lower()}.png' for brand_name in self.brands_mapping.values()}

        # Set up the temp directory
        self.current_script_dir = os.path.dirname(__file__)
        self.temp_dir = os.path.join(self.current_script_dir, '..', '.eng_temp')
        self.cut_audio_dir = os.path.join(self.temp_dir, 'cut_audio')

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        os.makedirs(self.cut_audio_dir)

    def __process_loop(self):
        while self.continue_flag:
            print("Processing")
            # Record 5 seconds of audio, then save it
            recording, num_channels = self.microphone.record(RECORDING_LENGTH)
            start = time.time()
            discard_last_chunk = False

            # Wait for the recording to finish, making sure the GUI is still interactive
            while (time.time() - start) < RECORDING_LENGTH:
                time.sleep(0.1)
                QApplication.processEvents()

                # If the user clicked "Stop", this flag will be set to False
                if not self.continue_flag:
                    return
            # Recording should be finished, in which case this call will return immediately
            self.microphone.wait()
            print('Recording finished')

            # If we failed to get a recording, exit immediately
            if recording is None:
                break

            file_path = SoundRecorder.preprocess(recording, num_channels, 48000)

            # Our audio file is saved as output.wav in the current working directory
            # Feed the audio into the engine detector. If nothing is detected, don't
            # feed it into the neural network
            #if not self.detector.detect(os.path.join(self.temp_dir, 'output.wav')):
            #    print('No engine detected')
            #    continue
            #wprint('Engine detected!')

            # Extract the MFCCs from the file that was saved,
            # then feed them into the engine classifier
            self.mfcc_extractor.slice_audio(file_path, discard_last_chunk)
            mfccs = self.mfcc_extractor.extract()

            self.__classify(mfccs)
            QApplication.processEvents()


    def __handle_start(self):
        print('Start')
        self.continue_flag = True
        self.microphone.set_device(self.mic_menu.currentIndex())
        self.__process_loop()

    def __handle_stop(self):
        self.continue_flag = False
        print('Stopped')

    def __classify(self, mfccs):
        classification_results = self.classifier.classify(mfccs)

        results_accumulator = [0] * len(classification_results[0])

        # add the results for each class
        for result in classification_results:
            for index, value in enumerate(result):
                results_accumulator[index] += value

        # sorted_results = sorted(results_accumulator)
        sorted_results = sorted(range(len(results_accumulator)), key=lambda k: results_accumulator[k])
        brand_1 = self.brands_mapping[sorted_results[-1]]
        brand_2 = self.brands_mapping[sorted_results[-2]]
        brand_3 = self.brands_mapping[sorted_results[-3]]

        # Update the GUI images
        self.__update_image_1(brand_1)
        self.__update_image_2(brand_2)
        self.__update_image_3(brand_3)
        print("\nBRANDS")
        print(brand_1)
        print(brand_2)
        print(brand_3)

    def __setup_left_layout(self):
        overall_layout = QVBoxLayout()

        # Set up a tab-based widget for the user to select an audio source
        input_selection_box = QGroupBox('Input Selection')
        box_layout = QVBoxLayout()
        tabs = QTabWidget()
        mic_tab = self.__setup_mic_tab()
        file_tab = self.__setup_file_tab()
        tabs.addTab(mic_tab, 'Microphone')
        tabs.addTab(file_tab, 'Audio File')
        tabs.setFixedHeight(150)
        input_selection_box.setFixedHeight(200)
        box_layout.addWidget(tabs)
        input_selection_box.setLayout(box_layout)
        overall_layout.addWidget(input_selection_box)

        # Create a status bar for the program to show messages in
        status_layout = QHBoxLayout()
        status_label = QLabel('Status: ')
        self.status_box = QLineEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setText('Ready')
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_box)
        overall_layout.addLayout(status_layout)

        # Add buttons for the user to control the program with
        button_layout = QHBoxLayout()
        start_button = QPushButton('Start Analysis')
        stop_button = QPushButton('Stop Analysis')
        start_button.released.connect(self.__handle_start)
        stop_button.released.connect(self.__handle_stop)
        button_layout.addWidget(start_button)
        button_layout.addWidget(stop_button)
        overall_layout.addLayout(button_layout)

        return overall_layout

    def __setup_mic_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        mic_label = QLabel('Select Microphone:')

        # Create a dropdown menu with the available audio devices
        self.mic_menu = QComboBox()
        available_microphones = SoundRecorder.get_devices()
        self.mic_menu.addItems(available_microphones)

        # Add the GUI elements to the mic tab
        layout.addWidget(mic_label)
        layout.addWidget(self.mic_menu)

        tab.setLayout(layout)
        return tab

    def __setup_file_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.file_label = QLabel('File: No file selected')
        browse_button = QPushButton('Browse')
        browse_button.released.connect(self.__select_audio_file)
        browse_button.setFixedWidth(100)

        layout.addWidget(self.file_label)
        layout.addWidget(browse_button)

        tab.setLayout(layout)
        return tab

    def __select_audio_file(self):
        Tk().withdraw()
        self.selected_file = askopenfilename(defaultextension='wav')

        if len(self.selected_file) == 0:
            self.file_label.setText(f'File: No file selected')
            return

        basename = os.path.basename(self.selected_file)
        self.file_label.setText(f'File: {basename}')

    def __update_status(self, text: str):
        self.status_box.setText(text)

    def __setup_image_box_1(self):
        # Create a widget to contain the layout
        image_box_1 = QWidget()
        image_box_layout = QVBoxLayout(image_box_1)
        self.image_1_label = QLabel()
        self.image_1_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(250))
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
        self.image_2_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(250))
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
        self.image_3_label.setPixmap(QPixmap("photos/empty.png").scaledToWidth(250))
        self.image_3_description = QLabel("Brand 3: None")

        # Set alignment for the labels
        self.image_3_label.setAlignment(Qt.AlignCenter)
        self.image_3_description.setAlignment(Qt.AlignCenter)

        # Add widgets to image box layout
        image_box_layout.addWidget(self.image_3_label)
        image_box_layout.addWidget(self.image_3_description)

        return image_box_3  # Return the widget that contains the layout

    def __update_image_1(self, brand: str):
        self.image_1_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(250))
        self.image_1_description.setText(f'Brand 1: {brand}')

    def __update_image_2(self, brand: str):
        self.image_2_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(250))
        self.image_2_description.setText(f'Brand 2: {brand}')

    def __update_image_3(self, brand: str):
        self.image_3_label.setPixmap(QPixmap(self.brands_image_dict[brand]).scaledToWidth(250))
        self.image_3_description.setText(f'Brand 3: {brand}')
