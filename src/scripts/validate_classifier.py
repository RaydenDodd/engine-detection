import argparse
import math
import os
import pickle
import shutil

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from scripts import augment_segment_mfcc_extract_master
from scripts import train

# Global variables
SAMPLE_RATE = 48000
SEGMENT_LENGTH_MS = 100
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = SEGMENT_LENGTH_MS / 1000
EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)

def parse_cli():
    parser = argparse.ArgumentParser(description='Validates the engine classifier')
    parser.add_argument('--model', help='The Keras model to test (use absolute path)', default=None)
    parser.add_argument('--input_dirs', nargs='+',
                        help='List of input directories (DIR containing folders of brand names)(use absolute paths)',
                        default=None, required=True)
    parser.add_argument('--label_mapping', help='The pickle file containing the label-to-category mapping used during MFCC extraction', default=None)
    return parser.parse_args()


def setup(temp_directory):
    """
    Creates a new temporary directory for the script to stores its files in.
    If the directory already exists, it will be cleared of its contents
    :param temp_directory: An absolute path to where the temp directory should be created
    :return: None
    """
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    else:
        shutil.rmtree(temp_directory)
        os.makedirs(temp_directory)
        pass


def cleanup(temp_directory):
    """
    Deletes the temporary directory used by this script
    :param temp_directory: An absolute path to the script's temp directory
    :return: None
    """
    if os.path.exists(temp_directory):
        shutil.rmtree(temp_directory)

def extract_mfccs(file_path, temp_directory):
    # First, segment the file
    segment_path = os.path.join(temp_directory, 'test_segments')
    if os.path.exists(segment_path):
        shutil.rmtree(segment_path)
    os.makedirs(segment_path)

    audio = AudioSegment.from_file(file_path, 'wav')
    chunks = make_chunks(audio, SEGMENT_LENGTH_MS)
    for i, chunk in enumerate(chunks[:-1]):
        file_num = str(i).zfill(5)
        chunk.export(os.path.join(segment_path, f'{file_num}.wav'), format='wav')

    # For each audio chunk, extract the MFCCs and add it to the list
    mfccs = []
    for chunk in os.listdir(segment_path):
        signal, sr = librosa.load(os.path.join(segment_path, chunk), sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        if len(mfcc) > EXPECTED_NUM_MFCC_VECTORS:
            mfcc = mfcc[:EXPECTED_NUM_MFCC_VECTORS]
        elif len(mfcc) < EXPECTED_NUM_MFCC_VECTORS:
            padding = EXPECTED_NUM_MFCC_VECTORS - len(mfcc)
            mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

        mfccs.append(mfcc)

    return np.array(mfccs)

def main(script_dir, temp_directory, input_dirs, model, label_mapping):
    # These will help calculate accuracy stats
    files_processed = 0
    num_correct_top1 = 0
    num_correct_top3 = 0
    incorrect_predictions = []

    for audio_dir in input_dirs:
        i = 0
        dirs = None
        for root, dirs_temp, files in os.walk(audio_dir, topdown=True):
            # Get the list of directories in the audio input folder
            # On the first iteration, dirs_temp will contain the names of the brand folders
            if dirs is None and len(dirs_temp) != 0:
                dirs = dirs_temp
                continue
            else:
                dir_name = dirs[i]
                i += 1

            for file in files:  # Each file for the current brand
                # Extract MFCCs from the current file
                mfccs = extract_mfccs(os.path.join(root, file), temp_directory)
                # Applying the scaling seriously hinders performance for some reason
                #scaler = StandardScaler()
                #mfccs = scaler.fit_transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)

                # Run the model
                prediction = model.predict(mfccs, batch_size=EXPECTED_NUM_MFCC_VECTORS)
                prediction = np.sum(prediction, axis=0)
                sorted_predictions = sorted(range(len(prediction)), key=lambda k: prediction[k])

                top1 = label_mapping[sorted_predictions[-1]]
                top2 = label_mapping[sorted_predictions[-2]]
                top3 = label_mapping[sorted_predictions[-3]]

                if top1 == dir_name:
                    num_correct_top1 += 1
                    print(f'{file}: correct | Top prediction')
                    num_correct_top3 += 1
                elif top1 == dir_name or top2 == dir_name or top3 == dir_name:
                    num_correct_top3 += 1
                    print(f'{file}: correct | In top 3 predictions: {top1}, {top2}, {top3}')
                else:
                    incorrect_predictions.append(file)
                    print(f'{file}: incorrect | Not in top 3 predictions: {top1}, {top2}, {top3}')
                files_processed += 1
                print(f'Files processed: {files_processed}')

                pass

    print('Validation stats:')
    print(f"Top 1 validation accuracy: {num_correct_top1 / files_processed}")
    print(f'Top 3 validation accuracy: {num_correct_top3 / files_processed}')
    print(f'Incorrectly predicted files:\n{incorrect_predictions}')
    pass


if __name__ == '__main__':
    args = parse_cli()
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    if args.model is None:
        model_path = os.path.join(script_dir, '..', 'trained_models', 'engine_classify.keras')
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = args.model
        model = tf.keras.models.load_model(model_path)

    if args.label_mapping is None:
        label_mapping = os.path.join(script_dir, '..', 'trained_models', 'label_to_category_top_mapping.pickle')
    else:
        label_mapping = args.label_mapping

    temp_dir = os.path.join(script_dir, '..', '..', '.validation_temp')

    with open(label_mapping, 'rb') as pickle_file:
        class_number_to_brand = pickle.load(pickle_file)

    setup(temp_dir)

    main(script_dir, temp_dir, args.input_dirs, model, class_number_to_brand)

    cleanup(temp_dir)
