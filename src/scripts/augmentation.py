import os
import random
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-a', '--audiodir', type=str, help='The relative path to the directory containing folders with raw audio files')
    parser.add_argument('-c', '--augmentedoutputdir', type=str, help='The relative path to the directory that should contain folders with augmented audio files')
    return parser.parse_args()


def random_gain(audio_data, sample_rate):
    # Generate a random gain factor between 0.5 and 1.5
    gain = np.random.uniform(0.5, 1.5)

    # Apply the random gain to the audio data
    audio_data_with_gain = audio_data * gain

    # You can optionally clip the audio to ensure it stays within the valid range [-1, 1]
    audio_data_with_gain = np.clip(audio_data_with_gain, -1.0, 1.0)
    return audio_data_with_gain


def noise_addition(audio_data, sample_rate):
    # Generate random noise with the same length as the audio data
    noise = np.random.normal(0, 0.1, len(audio_data))  # Adjust the second parameter for noise intensity

    # Add the noise to the audio data
    audio_data_with_noise = audio_data + noise

    # You can optionally clip the audio to ensure it stays within the valid range [-1, 1]
    audio_data_with_noise = np.clip(audio_data_with_noise, -1.0, 1.0)
    return audio_data_with_noise


def time_stretching(audio_data, sample_rate):
    # Define the stretch factor (1.0 means no stretching)
    stretch_factor = random.uniform(1.1, 1.8)  # Adjust this value as needed

    # Apply time stretching
    audio_data_stretched = librosa.effects.time_stretch(audio_data, rate=stretch_factor)
    return audio_data_stretched


def pitch_shifting(audio_data, sample_rate):
    max_pitch_shift = 7
    min_pitch_shift = -7
    # Define the pitch shift amount in semitones (positive value for increase, negative for decrease)
    pitch_shift_amount = random.uniform(min_pitch_shift, max_pitch_shift)  # Adjust this value as needed

    # Apply pitch scaling
    audio_data_pitch_shifted = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_shift_amount)
    return audio_data_pitch_shifted


def highpass_filter(audio_data, sample_rate):
    order = 5
    # Apply high-pass filtering with cutoff frequency 20 Hz to 20 kHz
    min_cutoff = 1000
    max_cutoff = 2000
    cutoff_freq = random.uniform(min_cutoff, max_cutoff)  # Adjust this value as needed

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    filtered_audio_data = filtfilt(b, a, audio_data)
    return filtered_audio_data


def lowpass_filter(audio_data, sample_rate):
    order = 5
    # Apply high-pass filtering with cutoff frequency 20 Hz to 20 kHz
    min_cutoff = 1000
    max_cutoff = 2000
    cutoff_freq = random.uniform(min_cutoff, max_cutoff)  # Adjust this value as needed
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    filtered_audio_data = filtfilt(b, a, audio_data)
    return filtered_audio_data


def apply_augmentation():
    functions = [random_gain, noise_addition, time_stretching, highpass_filter, lowpass_filter]
    sample_rate = 48000

    for i, brand in enumerate(augment_output_dirs):
        brand_files = os.listdir(augment_output_dirs[brand])
        print('Augmenting {} audio files...'.format(brand))

        # List all audio files in the folder
        audio_files = [file for file in os.listdir(augment_output_dirs[brand]) if file.endswith(('.wav', '.mp3', '.ogg', '.flac', '.WAV'))]

        # Calculate the number of samples to create for each audio file
        samples_per_file = 10 // len(audio_files)

        # Loop over each audio file in the brand folder
        for file in brand_files:
            if file.endswith(".mp3") or file.endswith(".wav") or file.endswith(".ogg") or file.endswith(".WAV"):  # Add more extensions if needed
                # Load the audio file
                file_path = r'{}/{}'.format(augment_output_dirs[brand], file)
                signal, sr = librosa.load(file_path, sr=sample_rate)

                output_folder = r'{}/augmented_{}_files'.format(augment_output_dirs[brand], brand)
                print(output_folder)
                # Check if the folder exists
                if not os.path.exists(output_folder):
                    # If the folder doesn't exist, create it
                    os.makedirs(output_folder)

                for j in range(samples_per_file):
                    # Define the number of augmentations to apply (randomly between 1 and 3)
                    num_augmentations = random.randint(1, 3)

                    # Randomly select augmentation functions
                    selected_augmentations = random.sample(functions, num_augmentations)

                    # Apply selected augmentations to the audio file
                    augmented_audio_data = signal  # Initialize with original audio data
                    for augmentation_function in selected_augmentations:
                        augmented_audio_data = augmentation_function(augmented_audio_data, sr)

                    output_file_path = r'{}/{}_{}_{}.wav'.format(output_folder, os.path.splitext(file)[0], '_'.join(func.__name__ for func in selected_augmentations),j)
                    # Save the augmented audio data to a new file using soundfile
                    print('Saving to: ' + output_file_path)
                    sf.write(output_file_path, augmented_audio_data, sample_rate)


if __name__ == '__main__':
    args = parse_cli()
    audio_dir = args.audiodir
    augment_output_dir = args.augmentedoutputdir

    augment_output_dirs = {
        "Audi": r'audio/Audi'.format(augment_output_dir),
        "Chevrolet": r'audio/Chevrolet'.format(augment_output_dir),
        "Honda": r'audio/Honda'.format(augment_output_dir),
        "Hyundai": r'audio/Hyundai'.format(augment_output_dir),
        "Kia": r'audio/Kia'.format(augment_output_dir),
        "Lexus": r'audio/Lexus'.format(augment_output_dir),
        "Nissan": r'audio/Nissan'.format(augment_output_dir),
        "Scion": r'audio/Scion'.format(augment_output_dir),
        "Toyota": r'audio/Toyota'.format(augment_output_dir),
        "Volkswagen": r'audio/Volkswagen'.format(augment_output_dir)
    }

    apply_augmentation()


