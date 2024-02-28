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
        self.model = tf.keras.models.load_model('engine_detect.keras')

    def classify(self, mfccs):
        return self.model.predict(mfccs)

    @staticmethod
    def ordinal_to_class_vector(ordinal):
        class_vector = ordinal.argmax(axis=1)
        return class_vector
