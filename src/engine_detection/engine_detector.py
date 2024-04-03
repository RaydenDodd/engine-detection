import os
import math
import numpy as np
import librosa
from joblib import dump, load
SAMPLE_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLE_RATE / HOP_LENGTH)


class EngineDetector:

    def __init__(self):
        # Get the directory of the current script
        self.current_script_dir = os.path.dirname(__file__)
        
        # Construct the path to the joblib files
        model_path = os.path.join(self.current_script_dir, '..', 'trained_models', 'ExtraTrees_EngineDetection.joblib')
        scaler_path = os.path.join(self.current_script_dir, '..', 'trained_models', 'Scaler_EngineDetection.joblib')
        
        # Load the model and scaler using the full paths
        self.model = load(model_path)
        self.scaler = load(scaler_path)

    # Extract Audio Features
    def extract_features(self, audio_path):
        signal, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        return mfcc

    # Return true if file is detected as an engine
    def detect(self, filename):
            
        file_path = os.path.join(self.current_script_dir, filename)
        print(file_path)
        # Test a file not used in training or testing
        mfcc = self.extract_features(file_path)

        #predictions = []

        #for mfcc in mfccs:
        #    reshaped = mfcc.reshape(1, -1)
        #    test_features = self.scaler.transform(reshaped)
        #    predictions.append(self.model.predict(test_features))

        #return 1 in predictions

        # Calculate mean MFCC
        mean_mfcc = np.mean(mfcc, axis=0)

        # Correcting the shape for scaler.transform
        mean_mfcc_reshaped = mean_mfcc.reshape(1, -1)

        # Use the reshaped array
        test_features = self.scaler.transform(mean_mfcc_reshaped)

        # Predict the class for the new audio
        prediction = self.model.predict(test_features)
        return bool(prediction)


# detector = EngineDetector()
# result = detector.detect("2006_Toyota_camry_phone_recording.m4a_segment_5.WAV")
# print(f"Engine Detected: {result}")
