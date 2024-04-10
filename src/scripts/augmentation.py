import os
import random
import argparse
import numpy as np
import librosa
import audiomentations
import soundfile as sf
from scipy.signal import butter, filtfilt


def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-a', '--audiodir', type=str, help='The relative path to the directory containing folders with raw audio files')
    parser.add_argument('-c', '--augmentedoutputdir', type=str, help='The relative path to the directory that should contain folders with augmented audio files')
    return parser.parse_args()


def random_gain(audio_data, sample_rate):
    # Generate a random gain factor between 0.01 and 0.3
    gain = np.random.uniform(0.01, 0.3)

    # Apply the random gain to the audio data
    audio_data_with_gain = audio_data * gain

    # You can optionally clip the audio to ensure it stays within the valid range [-1, 1]
    audio_data_with_gain = np.clip(audio_data_with_gain, -1.0, 1.0)
    return audio_data_with_gain


def noise_addition(audio_data, sample_rate):
    # Generate random noise with the same length as the audio data
    noise = np.random.normal(0.05, 0.2, len(audio_data))  # Adjust the second parameter for noise intensity

    # Add the noise to the audio data
    audio_data_with_noise = audio_data.transpose() + noise

    # You can optionally clip the audio to ensure it stays within the valid range [-1, 1]
    audio_data_with_noise = np.clip(audio_data_with_noise, -1.0, 1.0)
    return audio_data_with_noise.transpose()


def time_stretching(audio_data, sample_rate):
    # Define the stretch factor (1.0 means no stretching)
    stretch_factor = random.uniform(1.05, 1.5)  # Adjust this value as needed

    # Apply time stretching
    audio_data_stretched = librosa.effects.time_stretch(audio_data, rate=stretch_factor)

    return audio_data_stretched


def pitch_shifting(audio_data, sample_rate):
    max_pitch_shift = 5
    min_pitch_shift = 0.1
    # Define the pitch shift amount in semitones (positive value for increase, negative for decrease)
    pitch_shift_amount = random.uniform(min_pitch_shift, max_pitch_shift)  # Adjust this value as needed

    # Apply pitch scaling
    audio_data_pitch_shifted = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_shift_amount)
    return audio_data_pitch_shifted


def highpass_filter(audio_data, sample_rate):
    order = 5
    # Apply high-pass filtering with cutoff frequency 20 Hz to 20 kHz
    min_cutoff = 1000
    max_cutoff = 2500
    cutoff_freq = random.uniform(min_cutoff, max_cutoff)  # Adjust this value as needed

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    filtered_audio_data = filtfilt(b, a, audio_data)
    return filtered_audio_data


def lowpass_filter(audio_data, sample_rate):
    order = 5
    nyquist = 0.5 * sample_rate
    # Set max cutoff to 95% of the Nyquist frequency to ensure it's always valid
    max_cutoff = 0.95 * nyquist
    min_cutoff = 18000  # You might still need to adjust this based on your needs

    # Ensure min_cutoff does not exceed max_cutoff
    min_cutoff = min(min_cutoff, max_cutoff)

    cutoff_freq = random.uniform(min_cutoff, max_cutoff)
    normalized_cutoff = cutoff_freq / nyquist

    # if not 0 < normalized_cutoff < 1:
    #     print(f"Adjusted normalized lowpass cutoff frequency: {normalized_cutoff}")
    #     return audio_data  # You might want to handle this case differently

    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_audio_data = filtfilt(b, a, audio_data)
    return filtered_audio_data


def apply_augmentation():
    functions = [random_gain, noise_addition, pitch_shifting, highpass_filter, lowpass_filter]
    sample_rate = 48000

    for i, brand in enumerate(input_dirs):
        brand_files = os.listdir(input_dirs[brand])
        print('Augmenting {} audio files...'.format(brand))

        # List all audio files in the folder
        audio_files = [file for file in os.listdir(input_dirs[brand]) if file.endswith(('.wav', '.mp3', '.ogg', '.flac', '.WAV'))]

        # IMPORTANT: Change number of audio files to calculate the number of samples to create for each audio file
        samples_per_file = 5

        # Loop over each audio file in the brand folder
        for file in brand_files:
            if file.endswith(".mp3") or file.endswith(".wav") or file.endswith(".ogg") or file.endswith(".WAV"):  # Add more extensions if needed
                # Load the audio file
                file_path = r'{}/{}'.format(input_dirs[brand], file)
                signal, sr = librosa.load(file_path, sr=sample_rate, mono=False)

                output_folder = r'{}/augmented_{}_files'.format(output_dirs[brand], brand)
                print(output_folder)
                # Check if the folder exists
                if not os.path.exists(output_folder):
                    # If the folder doesn't exist, create it
                    os.makedirs(output_folder)

                for j, augmentation_function in enumerate(functions):
                    try:
                        augmented_audio_data = augmentation_function(signal, sample_rate)

                        output_file_path = r'{}/{}_{}_{}.WAV'.format(output_folder, os.path.splitext(file)[0], augmentation_function.__name__, j)
                        # Save the augmented audio data to a new file using soundfile
                        print('Saving to: ' + output_file_path)
                        sf.write(output_file_path, augmented_audio_data.transpose(), sample_rate, subtype='FLOAT')
                    except Exception as e:
                        # Handle the error gracefully
                        print(f"Error occurred while applying augmentation function {j}: {e}")
                        continue  # Skip to the next iteration


if __name__ == '__main__':
    args = parse_cli()
    audio_dir = args.audiodir
    augment_output_dir = args.augmentedoutputdir

    # Modified dirs for top 10 car brands only
    input_dirs = {
        'BMW': os.path.join(audio_dir, 'BMW'),
        'Ford': os.path.join(audio_dir, 'Ford'),
        'GMC': os.path.join(audio_dir, 'GMC'),
        'Honda': os.path.join(audio_dir, 'Honda'),
        'Hyundai': os.path.join(audio_dir, 'Hyundai'),
        'Jeep': os.path.join(audio_dir, 'Jeep'),
        'Kia': os.path.join(audio_dir, 'Kia'),
        'Nissan': os.path.join(audio_dir, 'Nissan'),
        'Subaru': os.path.join(audio_dir, 'Subaru'),
        'Toyota': os.path.join(audio_dir, 'Toyota')
    }

    output_dirs = {
        'BMW': os.path.join(augment_output_dir, 'BMW'),
        'Ford': os.path.join(augment_output_dir, 'Ford'),
        'GMC': os.path.join(augment_output_dir, 'GMC'),
        'Honda': os.path.join(augment_output_dir, 'Honda'),
        'Hyundai': os.path.join(augment_output_dir, 'Hyundai'),
        'Jeep': os.path.join(augment_output_dir, 'Jeep'),
        'Kia': os.path.join(augment_output_dir, 'Kia'),
        'Nissan': os.path.join(augment_output_dir, 'Nissan'),
        'Subaru': os.path.join(augment_output_dir, 'Subaru'),
        'Toyota': os.path.join(augment_output_dir, 'Toyota')
    }

    # augment_output_dirs = {
    #     "Audi": r'audio/Audi'.format(augment_output_dir),
    #     "BMW": r'audio/BMW'.format(augment_output_dir),
    #     "Chevrolet": r'audio/Chevrolet'.format(augment_output_dir),
    #     "Chevrolet (GMC)": r'audio/Chevrolet (GMC)'.format(augment_output_dir),
    #     "Chrysler (Jeep)": r'audio/Chrysler (Jeep)'.format(augment_output_dir),
    #     "Ford": r'audio/Ford'.format(augment_output_dir),
    #     "GMC": r'audio/GMC'.format(augment_output_dir),
    #     "Honda": r'audio/Honda'.format(augment_output_dir),
    #     "Hyundai": r'audio/Hyundai'.format(augment_output_dir),
    #     "Infiniti (Nissan)": r'audio/Infiniti (Nissan)'.format(augment_output_dir),
    #     "Jeep": r'audio/Jeep'.format(augment_output_dir),
    #     "Kia": r'audio/Kia'.format(augment_output_dir),
    #     "Lexus": r'audio/Lexus'.format(augment_output_dir),
    #     "Mazda": r'audio/Mazda'.format(augment_output_dir),
    #     "Mercedes-Benz": r'audio/Mercedes-Benz'.format(augment_output_dir),
    #     "Mitsubishi": r'audio/Mitsubishi'.format(augment_output_dir),
    #     "Nissan": r'audio/Nissan'.format(augment_output_dir),
    #     "Saturn": r'audio/Saturn'.format(augment_output_dir),
    #     "Scion": r'audio/Scion'.format(augment_output_dir),
    #     "Scion (Toyota)": r'audio/Scion (Toyota)'.format(augment_output_dir),
    #     "Subaru": r'audio/Subaru'.format(augment_output_dir),
    #     "Suzuki": r'audio/Suzuki'.format(augment_output_dir),
    #     "Toyota": r'audio/Toyota'.format(augment_output_dir),
    #     "Volkswagen": r'audio/Volkswagen'.format(augment_output_dir),
    #     "Volvo": r'audio/Volvo'.format(augment_output_dir),
    # }

    apply_augmentation()


