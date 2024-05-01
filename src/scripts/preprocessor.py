import sounddevice as sd
import librosa
from pydub import AudioSegment
from scipy.io.wavfile import write

DURATION = 5  # Recording 5-second audio chunks
TARGET_SAMPLE_RATE = 48000
TARGET_NUM_CHANNELS = 2


def get_mic_audio(mic_index, num_channels, sample_rate):
    sd.default.device = mic_index
    recording = sd.rec(int(DURATION * sample_rate), samplerate=sample_rate, channels=num_channels)
    sd.wait()
    return recording


def resample(audio, orig_sample_rate):
    return librosa.resample(audio, orig_sr=orig_sample_rate, target_sr=TARGET_SAMPLE_RATE)


def preprocess(audio, num_channels, sample_rate):
    data_type = 'ndarray'

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    if num_channels != TARGET_NUM_CHANNELS:
        write('temp.wav', TARGET_SAMPLE_RATE, audio)
        left = AudioSegment.from_wav('temp.wav')
        right = AudioSegment.from_wav('temp.wav')
        audio = AudioSegment.from_mono_audiosegments(left, right)
        data_type = 'audiosegment'

    # Write the processed audio to a file
    if data_type == 'ndarray':
        write('output.wav', TARGET_SAMPLE_RATE, audio)
    else:
        audio.export('output.wav', format='wav')


if __name__ == '__main__':
    # List the microphones connected to the system
    print(sd.query_devices())

    # Manually selected
    device_id = 6
    device_info = sd.query_devices(device=device_id)
    channels = device_info['max_input_channels']
    samplerate = int(device_info['default_samplerate'])

    # Get audio from the selected device
    raw_audio = get_mic_audio(device_id, TARGET_NUM_CHANNELS, TARGET_SAMPLE_RATE)

    preprocess(raw_audio, channels, TARGET_SAMPLE_RATE)
