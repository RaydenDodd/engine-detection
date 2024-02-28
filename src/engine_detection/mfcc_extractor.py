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
        pass

    def slice_audio(self):
        file_no_ext = re.split(r'.wav', 'output.wav')[0]

        # Load the audio file and turn it into chunks
        audio = AudioSegment.from_file('eng_temp/output.wav', AUDIO_FILE_EXT)
        audio_chunks = make_chunks(audio, AUDIO_CHUNK_LEN)

        # Save the chunks to their own output files
        # The last chunk may not be the full length, so we discard it
        for i, chunk in enumerate(audio_chunks[:-1]):
            file_num = str(i).zfill(5)
            chunk_name = r'{}_{}.{}'.format(file_no_ext, file_num, AUDIO_FILE_EXT)

            output_relative = r'eng_temp/cut_audio/{}'.format(chunk_name)
            chunk.export(output_relative, format=AUDIO_FILE_EXT)

    def extract(self):
        mfccs = []
        sample_rate = 48000
        duration = 5
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512
        #expected_num_mfcc_vectors = math.ceil(sample_rate / hop_length) * 5 - 1
        expected_num_mfcc_vectors = math.ceil(sample_rate / hop_length)

        files = os.listdir(r'eng_temp/cut_audio')

        for file in files:

            # Load the audio file
            file_path = r'eng_temp/cut_audio/{}'.format(file)
            signal, sr = librosa.load(file_path, sr=sample_rate)

            # Extract the MFCCs from the file
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
            mfcc = mfcc.T.tolist()

            # Ensure that all snippets have the same length
            if len(mfcc) == expected_num_mfcc_vectors:
                mfccs.append(mfcc)
            else:
                raise ValueError('Length of MFCC vectors didn\'t match expected value')

        return mfccs



