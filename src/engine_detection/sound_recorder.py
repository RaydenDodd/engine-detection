import os

import sounddevice
import sounddevice as sd
import librosa
from pydub import AudioSegment
from scipy.io.wavfile import write

SAMPLE_RATE = 48000
NUM_CHANNELS = 2


class SoundRecorder:
    __script_path = os.path.abspath(__file__)
    __script_dir = os.path.dirname(__script_path)
    __eng_temp_dir = os.path.join(__script_dir, '..', '.eng_temp')

    def __init__(self):
        self.device_id = -1

    def set_device(self, index: int):
        if index < 0 or index >= len(sd.query_devices()):
            raise IndexError('Device index out of range')
        else:
            self.device_id = index

    def record(self, duration: int):
        if self.device_id == -1:
            raise ValueError('Device index not set')

        sd.default.device = self.device_id
        try:
            recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=NUM_CHANNELS)
            num_channels = NUM_CHANNELS
        except sounddevice.PortAudioError:  # Microphone couldn't record in 2-channel mode
            try:
                recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
                num_channels = 1
            except sounddevice.PortAudioError:
                print('Could not record from audio device')
                return None, None

        sd.wait()
        return recording, num_channels

    @staticmethod
    def get_devices() -> list[str]:
        devices = sd.query_devices()
        names = list()

        for device in devices:
            names.append(device['name'])

        return names

    @staticmethod
    def preprocess(audio, num_channels, sample_rate):
        output_path = os.path.join(SoundRecorder.__eng_temp_dir, 'processed_audio.wav')
        if os.path.exists(output_path):
            os.remove(output_path)

        data_type = 'ndarray'

        if sample_rate != SAMPLE_RATE:
            audio = SoundRecorder.__resample(audio, sample_rate)

        if num_channels != NUM_CHANNELS:
            temp_path = os.path.join(SoundRecorder.__eng_temp_dir, 'temp.wav')
            write(temp_path, SAMPLE_RATE, audio)
            left = AudioSegment.from_wav(temp_path)
            right = AudioSegment.from_wav(temp_path)
            audio = AudioSegment.from_mono_audiosegments(left, right)
            os.remove(temp_path)
            data_type = 'audiosegment'

        # Write the processed audio to a file
        if data_type == 'ndarray':
            write(output_path, SAMPLE_RATE, audio)
        else:
            audio.export(output_path, format='wav')

        return output_path

    @staticmethod
    def __resample(audio, orig_sample_rate):
        return librosa.resample(audio, orig_sr=orig_sample_rate, target_sr=SAMPLE_RATE)
