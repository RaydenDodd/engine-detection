from pydub import AudioSegment
from pathlib import Path

OUT_FORMAT = 'mp3'
OUT_BITRATE = '512k'


def compress(input_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Create the output file name
    out_file = fr'{Path(input_file).stem}.{OUT_FORMAT}'

    # Save the file with the new bitrate using mp3 compression
    audio.export(out_file, format=OUT_FORMAT, bitrate=OUT_BITRATE)


if __name__ == '__main__':
    compress('short.wav')
    compress('medium1.wav')
    compress('medium2.wav')
    compress('long.wav')
