import pandas as pd
import os
from pytube import YouTube
from pydub import AudioSegment


# Function to snip audio
def snip_audio(input_file, output_file, start_time, end_time):
    bit_depth = 32  # Bit depth
    sample_rate = 48000  # Sample rate in samples per second (Hz)
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Extract the portion you want
    snipped_audio = audio[start_time_ms:end_time_ms]

    # Set the desired parameters
    snipped_audio = snipped_audio.set_frame_rate(sample_rate)
    snipped_audio = snipped_audio.set_channels(2)  # Stereo audio
    snipped_audio = snipped_audio.set_sample_width(32//8)  # 2 bytes per sample

    # Export the snipped audio to a new file
    snipped_audio.export(output_file, format="wav")  # Adjust format as needed


# Read the Excel file into a pandas DataFrame
df = pd.read_excel('src\scripts\Engine_Sounds.xlsx')  # Replace "input_data.xlsx" with the path to your Excel file

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Get the YouTube video URL from the current row
    video_url = row["Link 2"]  # Replace "YouTube_URL_Column_Name" with the actual column name containing the URLs
    model_name = row["Non car engine sound"]
    # brand_name = row["Brand"]
    # year = row["year"]
    start_time_sec = row["Begin 2"]
    end_time_sec = row["End 2"]

    # Convert seconds to milliseconds
    start_time_ms = start_time_sec * 1000
    end_time_ms = end_time_sec * 1000

    # Create a YouTube object
    yt = YouTube(video_url)

    # Get the highest resolution audio stream
    audio_stream = yt.streams.filter(only_audio=True).first()

    # folder_path = f"./audio/{brand_name}"
    folder_path = 'src/scripts/None/Other'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # audio_stream.download(output_path=folder_path, filename=f"{year}_{brand_name}_{model_name}_{index}.wav")
    audio_stream.download(output_path=folder_path, filename=f"{model_name}_{index}.wav")
    print(f"Audio {index} downloaded successfully.")

    file_path = f'src/scripts/None/Other/{model_name}_{index}.wav'
    print(file_path)
    # Snip the audio
    snip_audio(file_path, file_path, start_time_sec, end_time_sec)
    print(f"Audio {index} snipped successfully.")
