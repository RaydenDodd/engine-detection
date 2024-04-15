import os
import math
import numpy as np
import librosa
from joblib import dump, load
from collections import deque

SAMPLE_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 1 
TIMES_IN_A_ROW = 5
EXPECTED_NUM_MFCC_VECTORS = math.ceil((SAMPLE_RATE * DURATION) / HOP_LENGTH)


class EngineDetector:

    def __init__(self):
        # Get the directory of the current script
        self.current_script_dir = os.path.dirname(__file__)
        
        # Construct the path to the joblib files
        model_path = os.path.join(self.current_script_dir, '..', 'trained_models', 'RandomForest_pipeline.joblib')
        
        # Load the model and scaler using the full paths
        self.model_pipeline = load(model_path)

        # Initialize a deque with a maximum length of 5 to store the last 5 predictions
        self.last_predictions = deque(maxlen=TIMES_IN_A_ROW)

    # Extract Audio Features
    @staticmethod
    def extract_features(audio_path):
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T  # Transpose for model input
        if len(mfcc) > EXPECTED_NUM_MFCC_VECTORS:
            mfcc = mfcc[:EXPECTED_NUM_MFCC_VECTORS]
        elif len(mfcc) < EXPECTED_NUM_MFCC_VECTORS:
            padding = EXPECTED_NUM_MFCC_VECTORS - len(mfcc)
            mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

        return mfcc

    # Return true if file is detected as an engine
    def detect(self, filename):
        file_path = os.path.join(self.current_script_dir, filename)
        # Test a file not used in training or testing
        mfcc = self.extract_features(file_path)

        # Calculate mean MFCC
        mean_mfcc = np.mean(mfcc, axis=0)

        # Correcting the shape for scaler.transform
        mean_mfcc_reshaped = mean_mfcc.reshape(1, -1)

        # Predict the class for the new audio
        prediction = self.model_pipeline.predict(mean_mfcc_reshaped)[0]  # Ensure to get the first item in prediction array

        # Engine detected = 1, no engine detected = 0
        return bool(prediction)

        # The following code doesn't work unless the program has been running long enough
        # to fill the deque
        # Update the deque with the new prediction
        #self.last_predictions.append(prediction)

        # Check if the deque is full and all values are 1
        #return len(self.last_predictions) == TIMES_IN_A_ROW and all(pred == 1 for pred in self.last_predictions)


