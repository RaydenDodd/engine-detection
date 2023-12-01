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

AUDIO_CHUNK_LEN = 1000  # in milliseconds
AUDIO_FILE_EXT = 'wav'

def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-i', '--input', type=str,
                        help='The input file to classify')
    return parser.parse_args()

def slice_audio(file):

    file_no_ext = re.split(r'.wav', file)[0]

    # Load the audio file and turn it into chunks
    audio = AudioSegment.from_file(input_file, AUDIO_FILE_EXT)
    audio_chunks = make_chunks(audio, AUDIO_CHUNK_LEN)

    # Save the chunks to their own output files
    # The last chunk may not be the full length, so we discard it
    for i, chunk in enumerate(audio_chunks[:-1]):
        file_num = str(i).zfill(5)
        chunk_name = r'{}_{}.{}'.format(file_no_ext, file_num, AUDIO_FILE_EXT)

        output_relative = r'temp/{}'.format(chunk_name)
        chunk.export(output_relative, format=AUDIO_FILE_EXT)


def extract_mfccs():
    data = {
        "train": {"mfcc": [], "labels": [], "category": []},
        "test": {"mfcc": [], "labels": [], "category": []},
    }
    mfccs = []

    sample_rate = 48000
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    expected_num_mfcc_vectors = math.ceil(sample_rate / hop_length)

    files = os.listdir(r'temp')

    # Loop over each audio file in the brand folder
    for file in files:
        # Get the file name without the extension
        # file_no_ext = re.split(r'.' + AUDIO_FILE_EXT, file)[0]

        # Load the audio file
        file_path = r'temp/{}'.format(file)
        signal, sr = librosa.load(file_path, sr=sample_rate)

        # Extract the MFCCs from the file
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T.tolist()

        # Ensure that all snippets have the same length
        if len(mfcc) == expected_num_mfcc_vectors:
            mfccs.append(mfcc)

    return mfccs

def ordinal_to_class_vector(ordinal):
    class_vector = ordinal.argmax(axis=1)
    return class_vector


if __name__ == '__main__':
    if not os.path.exists(r'temp/'):
        os.makedirs(r'temp/')
    else:
        shutil.rmtree(r'temp/')
        os.makedirs(r'temp/')

    brands = ['Lexus', 'Nissan', 'Scion', 'Toyota']

    args = parse_cli()
    input_file = args.input

    slice_audio(input_file)
    mfccs = extract_mfccs()

    # Load the model
    model = tf.keras.models.load_model('engine_detect.keras')
    predictions = model.predict(mfccs)
    predictions = ordinal_to_class_vector(predictions)
    top_prediction = scipy.stats.mode(predictions)
    pred_brand = brands[top_prediction[0]]
    print(f'Predicted brand: {pred_brand}')
    pass
