import re
import shutil

import scipy
import tensorflow as tf
import numpy as np
import json
import argparse
import math
import os
import librosa
from librosa import feature
from pydub import AudioSegment
from pydub.utils import make_chunks

class EngineClassifier:

    def __init__(self):
        # Get the directory of the current script
        self.current_script_dir = os.path.dirname(__file__)
        model_path = os.path.join(self.current_script_dir, '..', 'trained_models', 'engine_detect_rayden_test_cnn_all_brands.keras')
        self.model = tf.keras.models.load_model(model_path)

    def classify(self, mfccs):
        return self.model.predict(mfccs)

    @staticmethod
    def ordinal_to_class_vector(ordinal):
        class_vector = ordinal.argmax(axis=1)
        return class_vector
