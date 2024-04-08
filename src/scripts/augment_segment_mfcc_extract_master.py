import argparse
from joblib import Parallel, delayed
import joblib
import os
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import math
import random
import pickle
#NOTE: YOU NEED TO pip install joblib librosa PySoundFile scipy pydub tqdm numpy pandas
 
#This file takes multiple input directorys that are strutrcued as folder of car brands
#This file augments the audio, segments all the audio, creates a dataframe out of the segments, and extracts mfccs and saves them for the top 10 brands and all brands
#NOTE: you can run this file multiple times it does not duplicate data it only does the augmentation and segmentation to new files in the input directory
#this file returns 4 things: an output_dir with all the segmented clips in their brands, a pandas dataframe with classes and filenames, 2 numpy arrays for features(features2d & features3d) and labels 


# Global variables
SAMPLE_RATE = 48000
SEGMENT_LENGTH_MS = 5000  # This can be overridden by command-line arguments
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = SEGMENT_LENGTH_MS / 1000  # Duration in seconds, this will update if SEGMENT_LENGTH_MS changes
EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)


top_10_brands_set = [
    "BMW",
    "Ford",
    "GMC",
    "Honda",
    "Hyundai",
    "Jeep",
    "Kia",
    "Nissan",
    "Subaru",
    "Toyota"
]

#Find your onedrive
########################################################################
def find_onedrive_virginia_tech_path():
    onedrive_envs = ['OneDriveCommercial', 'OneDriveConsumer']
    for env in onedrive_envs:
        onedrive_path = os.getenv(env)
        if onedrive_path and "Virginia Tech" in onedrive_path and os.path.isdir(onedrive_path):
            return onedrive_path
    
    home_path = os.path.expanduser('~')
    onedrive_vt_path = os.path.join(home_path, "OneDrive - Virginia Tech")
    if os.path.isdir(onedrive_vt_path):
        return onedrive_vt_path

    return None

def find_team_drive(onedrive_path, team_drive_name):
    # Construct the full path to the team drive
    team_drive_path = os.path.join(onedrive_path, team_drive_name)
    if os.path.isdir(team_drive_path):
        return team_drive_path
    else:
        return None




#Augmentation and Segmentation
########################################################################
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
    min_cutoff = 1000  # Hz
    max_cutoff = 2500  # Hz
    cutoff_freq = random.uniform(min_cutoff, max_cutoff)
    normalized_cutoff = cutoff_freq / (0.5 * sample_rate)
    # if not 0 < normalized_cutoff < 1:
    #     print(f"Invalid normalized lowpass cutoff frequency: {normalized_cutoff}.")
    #     print(f"Cutoff Frequency: {cutoff_freq} Hz")
    #     print(f"Sample Rate: {sample_rate} Hz")
    #     print(f"Nyquist Frequency: {0.5 * sample_rate} Hz")
    #     print(f"Audio Data Length (samples): {len(audio_data)}")
    #     print(f"Audio Data Duration (seconds): {len(audio_data)/sample_rate}")
    #     return audio_data
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
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

functions = [random_gain, noise_addition, pitch_shifting, highpass_filter, lowpass_filter]

# Process file function
def process_and_segment_file(file_path, output_dir, functions):
    filename = os.path.basename(file_path)
    base_filename, _ = os.path.splitext(filename)
    output_folder = os.path.join(output_dir, os.path.basename(os.path.dirname(file_path)))

    # Walk through the output directory to check if the base file has already been processed
    already_processed = False
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if base_filename in file:
                already_processed = True
                break
        if already_processed:
            break

    if already_processed:
        return {"path": file_path, "status": "Skipped", "error": "Already processed"}

    os.makedirs(output_folder, exist_ok=True)
    processed_info = {"path": file_path, "status": None, "error": None}
    try:    
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=None, mono=False)
        duration_in_ms = librosa.get_duration(y=signal, sr=sr) * 1000
        num_segments = int(np.floor(duration_in_ms / SEGMENT_LENGTH_MS))

        if num_segments == 0:
            error_message = "Error: File shorter than segment length"
            return {"path": file_path, "status": "Error", "error": error_message}


        for i in range(num_segments):
            start_sample = int((i * SEGMENT_LENGTH_MS) / 1000 * sr)
            end_sample = int(((i + 1) * SEGMENT_LENGTH_MS) / 1000 * sr)

            segment_signal = signal[:, start_sample:end_sample] if signal.ndim > 1 else signal[start_sample:end_sample]

            # Save the original segment
            segment_filename = f"{base_filename}_seg_{i+1}.wav"
            segment_path = os.path.join(output_folder, segment_filename)
            sf.write(segment_path, segment_signal.T, sr)  # .T for correct shape if stereo
            #print(f"Segment {i+1} saved: {segment_path}")

            # Apply augmentation functions to the segment
            for func_index, func in enumerate(functions, start=1):
                augmented_audio = func(segment_signal, sr)  # Apply the augmentation function
                func_name = func.__name__
                augmented_filename = f"{base_filename}_{func_name}_{func_index}_seg_{i + 1}.wav"
                augmented_path = os.path.join(output_folder, augmented_filename)
                sf.write(augmented_path, augmented_audio.T, sr)  # .T for correct shape if stereo
                #print(f"Augmented segment {i+1} ({func_name}) saved: {augmented_path}")

                
    except Exception as e:
         return {"path": file_path, "status": "Error", "error": str(e)}
    
    return {"path": file_path, "status": "Processed", "error": None}

def process_directories(input_dirs, output_dir, functions):
    print(f"Starting segmentation process for files in {input_dirs} to {output_dir} with segment length of {SEGMENT_LENGTH_MS}ms...\n")
    # Initialize a dictionary to hold the count of files per directory
    files_per_dir = {input_dir: 0 for input_dir in input_dirs}

    # Aggregate audio paths and count files per directory
    audio_paths = []
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            file_count = len(files)
            audio_paths.extend([os.path.join(root, file) for file in files])
            # Update the count for the input directory
            # Note: This assumes files directly under each input_dir are of interest. Adjust if counting in subdirectories is needed differently.
            files_per_dir[input_dir] += file_count

    # Print the file counts per input directory
    for dir_path, count in files_per_dir.items():
        print(f"Collected {count} files from {dir_path}")
    print(f"Done collecting all audio file paths: {len(audio_paths)}")

    if len(functions) != 0:
        print("\nBeginning Parallelized Augmentation and Segmentation")
    else:
        print("\nBeginning Parallelized Segmentation")
    processed_files = []
    skipped_files = []
    failed_files = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_and_segment_file, audio_path, output_dir, functions): audio_path for audio_path in audio_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            audio_path = futures[future]
            result = future.result()
            if result["status"] == "Processed":
                processed_files.append(audio_path)
            elif result["status"] == "Skipped":
                skipped_files.append(audio_path)
            elif result["status"] == "Error":
                failed_files.append((audio_path, result["error"]))

    print(f"\n\nOriginal files processed: {len(processed_files)}")
    # for file in processed_files:
    #     print(f"Processed: {file}")

    print(f"Original files skipped: {len(skipped_files)}")
    # for file in skipped_files:
    #     print(f"Skipped: {file}")

    if failed_files:
        print("Failed to process the following files:")
        for file, error in failed_files:
            print(f"{file}: {error}")

    print("Completed.")
    return processed_files

#DataFrame Creation
#######################################################################################################
# Function to process a single file
def process_file(folder_name, file_name, output_dir):
    return {
        "filename": file_name,
        "class": folder_name,
        "dir": os.path.join(output_dir, folder_name)
    }

# Function to process files in a given directory in parallel and create a DataFrame
def process_files_to_dataframe(path, output_dir):
    # Initialize an empty list to store futures
    futures_list = []
    df_entries = []

    with ThreadPoolExecutor() as executor:
        # Walk through the directory
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder_name}")
                # Create a future for each file
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        future = executor.submit(process_file, folder_name, file_name, output_dir)
                        futures_list.append(future)

        # Process the futures as they are completed
        for future in tqdm(as_completed(futures_list), total=len(futures_list), desc="Creating DataFrame"):
            result = future.result()
            df_entries.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(df_entries, columns=["filename", "class", "dir"])
    return df


#MFCC extraction
#############################################################################################
def extract_features(audio_path, sample_rate, n_mfcc, n_fft, hop_length, expected_num_mfcc_vectors):
    signal, sr = librosa.load(audio_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T

    if len(mfcc) > expected_num_mfcc_vectors:
        mfcc = mfcc[:expected_num_mfcc_vectors]
    elif len(mfcc) < expected_num_mfcc_vectors:
        padding = expected_num_mfcc_vectors - len(mfcc)
        mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

    return mfcc

def extract_features_parallel(data, sample_rate, n_mfcc, n_fft, hop_length, expected_num_mfcc_vectors):
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(extract_features_parallel_helper)(row, sample_rate, n_mfcc, n_fft, hop_length, expected_num_mfcc_vectors) for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting features")
    )
    return results

def extract_features_parallel_helper(row, sample_rate, n_mfcc, n_fft, hop_length, expected_num_mfcc_vectors):
    audio_path = os.path.join(row['dir'], row['filename'])
    if os.path.exists(audio_path):
        mfcc = extract_features(audio_path, sample_rate, n_mfcc, n_fft, hop_length, expected_num_mfcc_vectors)
        flattened_mfcc = np.mean(mfcc, axis=0)  # For 2D features
        return mfcc, flattened_mfcc, row['class']
    return None, None, None, None

def aggregate_features(results, set_of_brands=None):
    """
    Aggregate features into lists based on the set of brands.
    If set_of_brands is None, all features are aggregated.
    """
    # Create a dictionary to store the mapping of category names to labels
    category_to_label = {}
    label_counter = 0

    # Filter results based on set_of_brands if it's not None
    if set_of_brands is not None:
        filtered_results = [(mfcc_3d, mfcc_2d, category) for mfcc_3d, mfcc_2d, category in results if category in set_of_brands]
    else:
        filtered_results = results

    # Sort the filtered results by category (label) name alphabetically
    sorted_results = sorted(filtered_results, key=lambda x: x[2])

    features_3d, features_2d, labels, categorys = [], [], [], []

    for mfcc_3d, mfcc_2d, category in sorted_results:
        # Check if the category is already in the mapping
        if category not in category_to_label:
            # If the category is new, assign it the next label and increment the counter
            category_to_label[category] = label_counter
            label_counter += 1

        label = category_to_label[category]

        if mfcc_3d is not None:
            features_3d.append(mfcc_3d)
            features_2d.append(mfcc_2d)
            categorys.append(category)
            labels.append(label)

    # Invert the category_to_label dictionary to get a label_to_category mapping
    label_to_category = {v: k for k, v in category_to_label.items()}

    return np.array(features_3d), np.array(features_2d), np.array(labels), np.array(categorys), label_to_category


def save_features(features_2d, features_3d, labels, categorys, output_dir, filename, label_message):
        """
        Save the features and labels as compressed .npz file and print a completion message.

        Parameters:
        - features_2d: 2D array of features to be saved.
        - features_3d: 3D array of features to be saved.
        - labels: Array of labels corresponding to the features.
        - output_dir: Directory where the .npz file will be saved.
        - filename: Name of the .npz file without the extension.
        - label_message: Descriptive text for the print message indicating the type of data saved.
        """
        file_path = os.path.join(output_dir, f'{filename}.npz')
        np.savez_compressed(file_path, features_2d=features_2d, features_3d=features_3d, labels=labels, categorys=categorys)
        print(f"Completed saving features and labels for {label_message} at: {file_path}")

def save_mapping(folder_to_class_number, output_dir, filename):
    """
    Save the folder to class number mapping dictionary using pickle.

    Parameters:
    - folder_to_class_number: Dictionary containing the mapping of folder names to class numbers.
    - output_dir: Directory where the pickle file will be saved.
    - filename: Name of the pickle file without the extension.
    """
    file_path = os.path.join(output_dir, f'{filename}_mapping.pickle')
    
    with open(file_path, 'wb') as handle:
        pickle.dump(folder_to_class_number, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Completed saving the folder to class number mapping at: {file_path}")



def main(onedrive_enabaled, augmentation_enabled, dataframe_creation_enabled, feature_extraction_enabled, input_dirs, output_dir, mfcc_filename,mfcc_output_dir,segment_length_ms, n_mfcc, set_brands):
    global N_MFCC, SEGMENT_LENGTH_MS, DURATION, EXPECTED_NUM_MFCC_VECTORS
    N_MFCC=n_mfcc
    SEGMENT_LENGTH_MS = segment_length_ms
    DURATION = segment_length_ms / 1000  # Duration in seconds
    EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    
    if onedrive_enabaled:
        onedrive_vt_path = find_onedrive_virginia_tech_path()
        if onedrive_vt_path:
            print(f"OneDrive - Virginia Tech directory found at: {onedrive_vt_path}")
            team_drive_name = "General - S24-20_Engine Noise Detection"
            team_drive_path = find_team_drive(onedrive_vt_path, team_drive_name)
            if team_drive_path:
                print(f"Team drive '{team_drive_name}' found at: {team_drive_path}")
                os.chdir(team_drive_path)
                base_path = team_drive_path
            else:
                print(f"Team drive '{team_drive_name}' not found.")
                base_path = script_dir
        else:
            print("OneDrive - Virginia Tech directory not found.")
            base_path = script_dir
    else:
        base_path = script_dir
        print("OneDrive is not enabled. Using script directory.")
    print("Current directory:", os.getcwd())

    

    # Setup input and output directories based on base_path
    if input_dirs is None:
        input_dirs =  [os.path.join(team_drive_path, "audio"), os.path.join(team_drive_path, "online audio")]
    if output_dir is None:
        output_dir = os.path.join(team_drive_path, "segmented audio")

    # Setup mfcc_output_dir
    if mfcc_output_dir is None:
        mfcc_output_dir = os.path.join(script_dir, '..', 'trained_models')

        print("\n\ninput_dirs directory set to:", input_dirs)
        print("output_dir directory set to:", output_dir)
        print("MFCC output directory set to:", mfcc_output_dir)

    

    # Data Augmentation and Segmentation
    if augmentation_enabled:
        print("\n\n\nBeginning audio augmentation and segmentation")
        # Define your directories and call the function...
        functions = [random_gain, noise_addition, pitch_shifting, highpass_filter, lowpass_filter]

        processed_files = process_directories(input_dirs, output_dir, functions)
    else:
        print("\n\n\nSkipping audio augmentation\nBeginning audio segmentation")
        processed_files = process_directories(input_dirs, output_dir, [])
    


    if dataframe_creation_enabled :#and len(processed_files) > 0:
        print("\n\n\n Creating Dataframe")
        df= process_files_to_dataframe(output_dir, output_dir)

        # Print the counts for each class
        print(f"\nTotal size of DataFrame: {df.shape[0]}")
        for class_name in df['class'].unique():
            print(f"Count of class {class_name}: {df[df['class'] == class_name].shape[0]}")

        # Display the first few rows of the DataFrame
        print("\nHead of dataframe:")
        print(df.head())

        # Save the DataFrame to a CSV file
        df.to_csv("dataframe.csv", index=False)
    else:
        print("\n\n\nDid not create new dataframe as no new processed files.")


    # MFCC extraction
    if feature_extraction_enabled:
        print("\n\n\n Extracting MFCCs")
        if not dataframe_creation_enabled or not augmentation_enabled or len(processed_files) == 0:
            print("Loading csv")
            df = pd.read_csv('dataframe.csv')

        results = extract_features_parallel(df, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, EXPECTED_NUM_MFCC_VECTORS)

        if set_brands is None:
            features_3d_all, features_2d_all, labels_all, categorys_all, label_to_category_all = aggregate_features(results)
            features_3d_top, features_2d_top, labels_top, categorys_top, label_to_category_top = aggregate_features(results, top_10_brands_set)

            save_features(features_2d_all, features_3d_all, labels_all, categorys_all, mfcc_output_dir, f'{mfcc_filename}_all_brands', 'all brands')
            save_features(features_2d_top, features_3d_top, labels_top, categorys_top, mfcc_output_dir, f'{mfcc_filename}_top_10_brands', 'top 10 brands')

            save_mapping(label_to_category_all, mfcc_output_dir, 'label_to_category_all')
            save_mapping(label_to_category_top, mfcc_output_dir, 'label_to_category_top')

        else: 
            features_3d_set, features_2d_set, labels_set, categorys_set, label_to_category_set = aggregate_features(results, set_brands)

            save_features(features_2d_set, features_3d_set, labels_set, categorys_set, mfcc_output_dir, f'{mfcc_filename}_set_brands', 'the specified set of brands')
            save_mapping(label_to_category_set, mfcc_output_dir, 'label_to_category_set')
    else:
        print("\n\n\nSkipping MFCC Extraction")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation, dataframe creation, and feature extraction")
    parser.add_argument('--one_drive', help='Enable onedrive if you want the script to fnd your one drive path', action='store_true', default=True)
    parser.add_argument('--augment', help='Disable data augmentation and segmentation', action='store_false', default=True)
    parser.add_argument('--create_df', help='Disable dataframe creation', action='store_false', default=True)
    parser.add_argument('--extract_features', help='Disable feature extraction', action='store_false', default=True)
    parser.add_argument('--input_dirs', nargs='+', help='List of input directories (DIR containing folders of brand names)(use absolute paths)', default=None)
    parser.add_argument('--output_dir', help='Output directory for segments (Will create a DIR containing folders of the same brand names from input_dir)(Use absolute paths)', default=None)
    parser.add_argument('--mfcc_filename', help='Change the file name of the mfcc output', default=None)
    parser.add_argument('--mfcc_output_dir', help='Change where the mfcc output is going', default=None)
    parser.add_argument('--segment_length', type=int, help='Length of each audio segment in milliseconds', default=5000)
    parser.add_argument('--n_mfcc', type=int, help='Change the number of MFCC created', default=13)
    parser.add_argument('--set_brands', nargs='+', help='List of brands you want to include(instead of the top 10)', default=None)


    args = parser.parse_args()

    if args.mfcc_filename is None:
        args.mfcc_filename = 'MFCC'

    # Print the current settings
    print(f"Running script with these settings: \n"
          f"OneDrive: {'Enabled' if args.one_drive else 'Disabled'}\n"
          f"Data Augmentation: {'Enabled' if args.augment else 'Disabled'}\n"
          f"Dataframe Creation: {'Enabled' if args.create_df else 'Disabled'}\n"
          f"Feature Extraction: {'Enabled' if args.extract_features else 'Disabled'}\n"
          f"Mfcc Filename: {args.mfcc_filename}\n"
          f"N_MFCC: {args.n_mfcc}\n"
          f"Segment Length (ms): {args.segment_length}\n")

    if args.set_brands is not None:
        # Create a string representation of the selected brands if any, or a default message
        brands_str = ", ".join(args.set_brands) 
        print(f"Selected Brands: {brands_str}\n")

    main(
        onedrive_enabaled=args.one_drive,
        augmentation_enabled=args.augment,
        dataframe_creation_enabled=args.create_df,
        feature_extraction_enabled=args.extract_features,
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        mfcc_filename=args.mfcc_filename,
        mfcc_output_dir=args.mfcc_output_dir,
        n_mfcc=args.n_mfcc,
        segment_length_ms=args.segment_length,
        set_brands=args.set_brands
    )