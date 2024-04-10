import threading
from collections import deque
import math
import os
import numpy as np

SAMPLE_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 5  # seconds
EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)



# Change your path
####################################################
#change to something else
# path_to_directory = 'C:/Users/tigge/Documents/MDE'
# os.chdir(path_to_directory)
#where the .py file is
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

model_path = 'saved_models/DNN_bigset.keras'
model = tf.keras.models.load_model(model_path)
scaler = load('saved_models/scaler.joblib')
scaler_joblib = load('Scaler_EngineDetection.joblib')
model_joblib = load('ExtraTrees_EngineDetection.joblib')

#model_name = 'CNN'
#model_name = 'RNN'
#model_name = 'DNN'
model_name = 'ExtraTrees'

def extract_features(audio_clip):
    mfcc = librosa.feature.mfcc(y=audio_clip, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T  # Transpose to have frames as rows
    
    # Ensure MFCC features match the fixed_number_of_frames expected by the model
    # This might involve padding or truncating
    fixed_number_of_frames = EXPECTED_NUM_MFCC_VECTORS  # This should be set based on your model's training data
    if len(mfcc) > fixed_number_of_frames:
        # Truncate the sequence if it's longer than the desired number of frames
        mfcc = mfcc[:fixed_number_of_frames]
    elif len(mfcc) < fixed_number_of_frames:
        # Pad the sequence with zeros if it's shorter than the desired number of frames
        padding = fixed_number_of_frames - len(mfcc)
        mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
  
    return mfcc

def classify_audio_clip(model, mfcc, boolean):
    """
    Predict the class of an audio clip based on MFCC features.
    """
    prediction = model.predict(mfcc)
    if boolean:
        return 'Class 1' if prediction[0][0] > 0.5 else 'Class 0'
    else:
        return 'Class 1' if prediction[0] == 1 else 'Class 0'


class AudioBuffer:
    def __init__(self, duration, sample_rate):
        self.buffer = deque(maxlen=int(duration * sample_rate))

    def add(self, data):
        self.buffer.extend(data)

    def get_latest(self, duration, sample_rate):
        # Return the latest 'duration' seconds of audio
        num_samples = int(duration * sample_rate)
        if len(self.buffer) < num_samples:
            return np.zeros(num_samples)  # or handle this case differently
        return np.array(list(self.buffer)[-num_samples:])

audio_buffer = AudioBuffer(duration=5, sample_rate=SAMPLE_RATE)

def continuous_recording():
    while True:
        data = record_audio(duration=0.1, sample_rate=SAMPLE_RATE)  # Record shorter chunks
        audio_buffer.add(data)

def periodic_prediction():
    while True:
        latest_audio = audio_buffer.get_latest(duration=5, sample_rate=SAMPLE_RATE)
        mfcc_features = extract_features(latest_audio)
        # Continue with feature scaling and prediction as in your existing code
        # Print or process the prediction result

# Start the threads
recording_thread = threading.Thread(target=continuous_recording)
prediction_thread = threading.Thread(target=periodic_prediction)

recording_thread.start()
prediction_thread.start()
