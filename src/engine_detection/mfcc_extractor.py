import math
import os
import re

import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

AUDIO_CHUNK_LEN = 1000  # in milliseconds
AUDIO_FILE_EXT = 'wav'


class MFCCExtractor:

    def __init__(self):
        #need to find the path of where this script is stored so that we can work backwards to find eng_temp
        self.script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(self.script_path)
        # Define eng_temp directory path relative to the current script directory and one directory up
        self.eng_temp_dir = os.path.join(self.script_dir, '..', 'eng_temp')
        # Ensure the eng_temp and cut_audio directories exist
        os.makedirs(os.path.join(self.eng_temp_dir, 'cut_audio'), exist_ok=True)
        pass

    def slice_audio(self):
        file_no_ext = re.split(r'.wav', 'output.wav')[0]

        # Construct the path to the input audio file
        audio_file_path = os.path.join(self.eng_temp_dir, 'output.wav')

        # Load the audio file and turn it into chunks
        audio = AudioSegment.from_file(audio_file_path, AUDIO_FILE_EXT)
        audio_chunks = make_chunks(audio, AUDIO_CHUNK_LEN)

        # Save the chunks to their own output files
        # The last chunk may not be the full length, so we discard it
        for i, chunk in enumerate(audio_chunks[:-1]):
            file_num = str(i).zfill(5)
            chunk_name = r'{}_{}.{}'.format(file_no_ext, file_num, AUDIO_FILE_EXT)

            # Construct the output file path
            output_relative = os.path.join(self.eng_temp_dir, 'cut_audio', chunk_name)
            chunk.export(output_relative, format=AUDIO_FILE_EXT)

    def extract(self):
        #TODO: Why are we turning the 5 second clip into 1 second clips just to classify it on all 5??
        mfccs = []
        sample_rate = 48000
        duration = 5
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512
        duration = 5 # 5 seconds
        #expected_num_mfcc_vectors = math.ceil(sample_rate / hop_length) * 5 - 1
        expected_num_mfcc_vectors = math.ceil(sample_rate* duration / hop_length)

        # List files in the cut_audio directory
        # files = os.listdir(os.path.join(self.eng_temp_dir, 'cut_audio'))

        # for file in files:

        #     # Load the audio file
        #     # Construct the path to each file
        #     file_path = os.path.join(self.eng_temp_dir, 'cut_audio', file)
        #     signal, sr = librosa.load(file_path, sr=sample_rate)

        # Construct the path to the audio file
        audio_file_path = os.path.join(self.eng_temp_dir, 'output.wav')

        # Load the audio file
        signal, sr = librosa.load(audio_file_path, sr=sample_rate)

        # Extract the MFCCs from the file
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T.tolist()


        # Ensure that all snippets have the same length
        if len(mfcc) == expected_num_mfcc_vectors:
            mfccs.append(mfcc)
        else:
            raise ValueError('Length of MFCC vectors didn\'t match expected value')

        return mfccs



