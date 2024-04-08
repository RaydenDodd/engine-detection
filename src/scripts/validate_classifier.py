import argparse
import os

import scripts.train
from train import main

def parse_cli():
    parser = argparse.ArgumentParser(description='Performs LOOCV to validate a neural network')
    parser.add_argument('-m', '--mfccdir', type=str,
                        help='The relative path to the directory that should contain the npz file with the MFCCs', default=None)  # Corrected 'default'
    parser.add_argument('-mf', '--mfcc_filename', type=str,
                        help='The filename of the MFCC you want to train on', default=None)  # Corrected 'default' and minor grammar
    return parser.parse_args()

def main(script_dir, mfcc_dir, mfcc_filename):
    # Make a copy of the audio input firs

    # For each brand and for each file within that brand, remove the current file
    # Then extract MFCCs and train a model using the remaining audio clips
    # Extract MFCCs from the clip that was left out, then send it to the model for prediction
    # Check whether the prediction was correct and record the result

    #model = scripts.train.main(script_dir, mfcc_dir, mfcc_filename)
    pass

if __name__ == '__main__':
    args = parse_cli()
    if args.mfccdir is None:
        mfcc_dir = os.path.join('..', 'trained_models')
    else:
        mfcc_dir = args.mfccdir
    if args.mfcc_filename is None:
        mfcc_filename = 'MFCC_all_brands.npz'
    else:
        mfcc_filename = args.mfcc_filename

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    main(script_dir, mfcc_dir, mfcc_filename)