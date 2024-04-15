import math
import os
import re
import shutil

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

AUDIO_CHUNK_LEN_MS = 100  # in milliseconds
AUDIO_FILE_EXT = 'wav'
SAMPLE_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

class MFCCExtractor:

    def __init__(self):
        #need to find the path of where this script is stored so that we can work backwards to find eng_temp
        self.script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(self.script_path)
        # Define eng_temp directory path relative to the current script directory and one directory up
        self.eng_temp_dir = os.path.join(self.script_dir, '..', '.eng_temp')
        self.cut_audio_dir = os.path.join(self.eng_temp_dir, 'cut_audio')
        # Ensure the eng_temp and cut_audio directories exist
        os.makedirs(self.cut_audio_dir, exist_ok=True)
        pass

    def slice_audio(self, file_path: str):
        """
        Slices an audio file into equal-sized chunks whose length depends on AUDIO_CHUNK_LEN. Note:
        the file that gets sliced will not be modified.
        :param file_path: The absolute path to the audio file to slice
        :return: None
        """
        # Clear the output directory
        if os.path.exists(self.cut_audio_dir):
            shutil.rmtree(self.cut_audio_dir)
        os.makedirs(self.cut_audio_dir)

        file_no_ext = re.split(r'.wav', os.path.basename(file_path))[0]

        # Load the audio file and turn it into chunks
        audio = AudioSegment.from_file(file_path, AUDIO_FILE_EXT)
        audio_chunks = make_chunks(audio, AUDIO_CHUNK_LEN_MS)

        # Save the chunks to their own output files
        for i, chunk in enumerate(audio_chunks):
            file_num = str(i).zfill(5)
            chunk_name = r'{}_{}.{}'.format(file_no_ext, file_num, AUDIO_FILE_EXT)

            # Construct the output file path
            output_path = os.path.join(self.cut_audio_dir, chunk_name)
            chunk.export(output_path, format=AUDIO_FILE_EXT)

    def extract(self):
        mfccs = []
        expected_num_mfcc_vectors = math.ceil(SAMPLE_RATE * (AUDIO_CHUNK_LEN_MS / 1000) / HOP_LENGTH)

        # List files in the cut_audio directory
        files = os.listdir(self.cut_audio_dir)

        for file in files:
            # Load the audio file
            file_path = os.path.join(self.cut_audio_dir, file)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Extract the MFCCs from the file
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=N_FFT, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            mfcc = mfcc.T

            # Ensure that all snippets have the same length
            if len(mfcc) == expected_num_mfcc_vectors:
                mfccs.append(mfcc)
            else:
                raise ValueError("Length of MFCC vectors didn't match expected value")

        return np.array(mfccs)
