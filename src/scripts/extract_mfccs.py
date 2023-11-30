# extract_mfccs.py
# This script is used to extract Mel-Frequency Cepstral Coefficients from
# raw audio files. It first turns the audio files into 1-second chunks, then
# extracts MFCCs from those chunks

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

AUDIO_CHUNK_LEN = 1000  # in milliseconds
AUDIO_FILE_EXT = 'wav'  # input audio must be uncompressed .wav files


def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-a', '--audiodir', type=str, help='The relative path to the directory containing folders with raw audio files')
    parser.add_argument('-c', '--chunkoutputdir', type=str, help='The relative path to the directory that should contain folders with chunked audio files')
    parser.add_argument('-m', '--mfccoutputdir', type=str, help='The relative path to the directory that should contain folders with mfccs extracted from the chunked audio files')
    return parser.parse_args()


def slice_audio():
    # Check if the output dir is present and create it if needed
    if not os.path.exists(r'{}'.format(chunk_output_dir)):
        os.makedirs(r'{}'.format(chunk_output_dir))

    print('Splitting audio into chunks...')
    for brand in tqdm(raw_audio_dirs):
        # Check if the brand output directory exists and create it if needed
        if not os.path.exists(r'{}/{}'.format(chunk_output_dir, brand)):
            os.makedirs(r'{}/{}'.format(chunk_output_dir, brand))

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
                output_relative = r'{}/{}/{}'.format(chunk_output_dir, brand, chunk_name)
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

    # Loop over each brand in the cut audio folder
    for brand in tqdm(chunk_output_dirs):
        folder = chunk_output_dirs[brand]
        pass

if __name__ == '__main__':
    args = parse_cli()
    audio_dir = args.audiodir
    chunk_output_dir = args.chunkoutputdir
    mfcc_output_dir = args.mfccoutputdir

    raw_audio_dirs = {
        "Lexus": r'{}/Lexus'.format(audio_dir),
        "Nissan": r'{}/Nissan'.format(audio_dir),
        "Scion": r'{}/Scion'.format(audio_dir),
        "Toyota": r'{}/Toyota'.format(audio_dir)
    }

    chunk_output_dirs = {
        "Lexus": r'{}/Lexus'.format(chunk_output_dir),
        "Nissan": r'{}/Nissan'.format(chunk_output_dir),
        "Scion": r'{}/Scion'.format(chunk_output_dir),
        "Toyota": r'{}/Toyota'.format(chunk_output_dir)
    }

    mfcc_output_dirs = {
        "Lexus": r'{}/Lexus'.format(mfcc_output_dir),
        "Nissan": r'{}/Nissan'.format(mfcc_output_dir),
        "Scion": r'{}/Scion'.format(mfcc_output_dir),
        "Toyota": r'{}/Toyota'.format(mfcc_output_dir)
    }

    #slice_audio()
    extract_mfccs()
