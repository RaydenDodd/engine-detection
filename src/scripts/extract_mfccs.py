# extract_mfccs.py
# This script is used to extract Mel-Frequency Cepstral Coefficients from
# raw audio files. It first turns the audio files into 1-second chunks, then
# extracts MFCCs from those chunks
import random

# The procedures used in this script are inspired by those in
# https://becominghuman.ai/signal-processing-engine-sound-detection-a88a8fa48344

from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import re
from tqdm import tqdm
import json
import copy
import argparse
import os
import librosa
from librosa import feature

AUDIO_CHUNK_LEN = 1000  # in milliseconds
AUDIO_FILE_EXT = 'wav'  # input audio must be uncompressed .wav files
TRAIN_CHANCE = 0.8

def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-a', '--audiodir', type=str, help='The relative path to the directory containing folders with raw audio files')
    parser.add_argument('-c', '--chunkoutputdir', type=str, help='The relative path to the directory that should contain folders with chunked audio files')
    parser.add_argument('-m', '--mfccoutputdir', type=str, help='The relative path to the directory that should contain folders with mfccs extracted from the chunked audio files')
    return parser.parse_args()


def slice_audio():
    # Check if the output dirs are present and create them if needed
    if not os.path.exists(r'{}'.format(chunk_output_dir)):
        os.makedirs(r'{}'.format(chunk_output_dir))
    if not os.path.exists(r'{}'.format(chunk_train_output_dir)):
        os.makedirs(r'{}'.format(chunk_train_output_dir))
    if not os.path.exists(r'{}'.format(chunk_test_output_dir)):
        os.makedirs(r'{}'.format(chunk_test_output_dir))

    print('Splitting audio into chunks...')
    for brand in tqdm(raw_audio_dirs):
        # Check if the brand output directories exist and create them if needed
        if not os.path.exists(r'{}/{}'.format(chunk_train_output_dir, brand)):
            os.makedirs(r'{}/{}'.format(chunk_train_output_dir, brand))
        if not os.path.exists(r'{}/{}'.format(chunk_test_output_dir, brand)):
            os.makedirs(r'{}/{}'.format(chunk_test_output_dir, brand))

        # Get a list of all audio files for this brand
        brand_files = os.listdir(raw_audio_dirs[brand])

        for file in brand_files:
            # Get the relative path to the current file
            file_relative = r'{}/{}/{}'.format(audio_dir, brand, file)
            file_no_ext = re.split(r'.' + AUDIO_FILE_EXT, file)[0]

            # Load the audio file and turn it into chunks
            audio = AudioSegment.from_file(file_relative, AUDIO_FILE_EXT)
            audio_chunks = make_chunks(audio, AUDIO_CHUNK_LEN)

            # Save the chunks to their own output files
            # The last chunk may not be the full length, so we discard it
            for i, chunk in enumerate(audio_chunks[:-1]):
                file_num = str(i).zfill(5)
                chunk_name = r'{}_{}.{}'.format(file_no_ext, file_num, AUDIO_FILE_EXT)

                # Most files will go toward training while some will go toward testing
                if random.uniform(0, 1) < TRAIN_CHANCE:
                    output_relative = r'{}/{}'.format(chunk_train_output_dirs[brand], chunk_name)
                    chunk.export(output_relative, format=AUDIO_FILE_EXT)
                else:
                    output_relative = r'{}/{}'.format(chunk_test_output_dirs[brand], chunk_name)
                    chunk.export(output_relative, format=AUDIO_FILE_EXT)


def extract_mfccs():
    data = {
        "train": {"mfcc": [], "labels": [], "category": []},
        "test": {"mfcc": [], "labels": [], "category": []},
    }

    sample_rate = 48000
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    expected_num_mfcc_vectors = math.ceil(sample_rate / hop_length)

    # Loop over each brand in the cut audio train folder
    print('Extracting training MFCCs...')
    for brand in tqdm(chunk_train_output_dirs):
        brand_files = os.listdir(chunk_train_output_dirs[brand])

        # Loop over each audio file in the brand folder
        for i, file in enumerate(brand_files):
            # Get the file name without the extension
            #file_no_ext = re.split(r'.' + AUDIO_FILE_EXT, file)[0]

            # Load the audio file
            file_path = r'{}/{}'.format(chunk_train_output_dirs[brand], file)
            signal, sr = librosa.load(file_path, sr=sample_rate)

            # Extract the MFCCs from the file
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
            mfcc = mfcc.T.tolist()

            # Ensure that all snippets have the same length
            if len(mfcc) == expected_num_mfcc_vectors:
                data['train']['mfcc'].append(mfcc)
                data['train']['labels'].append(i)
                data['train']['category'].append(brand)

    # Loop over each brand in the cut audio train folder
    print('Extracting testing MFCCs...')
    for brand in tqdm(chunk_test_output_dirs):
        brand_files = os.listdir(chunk_test_output_dirs[brand])

        # Loop over each audio file in the brand folder
        for i, file in enumerate(brand_files):
            # Get the file name without the extension
            # file_no_ext = re.split(r'.' + AUDIO_FILE_EXT, file)[0]

            # Load the audio file
            file_path = r'{}/{}'.format(chunk_test_output_dirs[brand], file)
            signal, sr = librosa.load(file_path, sr=sample_rate)

            # Extract the MFCCs from the file
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
            mfcc = mfcc.T.tolist()

            # Ensure that all snippets have the same length
            if len(mfcc) == expected_num_mfcc_vectors:
                data['test']['mfcc'].append(mfcc)
                data['test']['labels'].append(i)
                data['test']['category'].append(brand)

    # Save the MFCC data
    # If the output directory doesn't exist, create it
    if not os.path.exists(r'{}'.format(mfcc_output_dir)):
        os.makedirs(r'{}'.format(mfcc_output_dir))

    with open(fr'{mfcc_output_dir}/mfccs.json', 'w') as mfcc_file:
        json.dump(data, mfcc_file, indent=4)

if __name__ == '__main__':
    args = parse_cli()
    audio_dir = args.audiodir
    chunk_output_dir = args.chunkoutputdir
    mfcc_output_dir = args.mfccoutputdir

    chunk_train_output_dir = fr'{chunk_output_dir}/train'
    chunk_test_output_dir = fr'{chunk_output_dir}/test'

    raw_audio_dirs = {
        "Lexus": r'{}/Lexus'.format(audio_dir),
        "Nissan": r'{}/Nissan'.format(audio_dir),
        "Scion": r'{}/Scion'.format(audio_dir),
        "Toyota": r'{}/Toyota'.format(audio_dir)
    }

    chunk_train_output_dirs = {
        "Lexus": r'{}/train/Lexus'.format(chunk_output_dir),
        "Nissan": r'{}/train/Nissan'.format(chunk_output_dir),
        "Scion": r'{}/train/Scion'.format(chunk_output_dir),
        "Toyota": r'{}/train/Toyota'.format(chunk_output_dir)
    }

    chunk_test_output_dirs = {
        "Lexus": r'{}/test/Lexus'.format(chunk_output_dir),
        "Nissan": r'{}/test/Nissan'.format(chunk_output_dir),
        "Scion": r'{}/test/Scion'.format(chunk_output_dir),
        "Toyota": r'{}/test/Toyota'.format(chunk_output_dir)
    }

    mfcc_output_dirs = {
        "Lexus": r'{}/Lexus'.format(mfcc_output_dir),
        "Nissan": r'{}/Nissan'.format(mfcc_output_dir),
        "Scion": r'{}/Scion'.format(mfcc_output_dir),
        "Toyota": r'{}/Toyota'.format(mfcc_output_dir)
    }

    slice_audio()
    extract_mfccs()
