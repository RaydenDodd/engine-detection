import os
from time import sleep

import sounddevice
import sounddevice as sd
import librosa
from pydub import AudioSegment
from scipy.io.wavfile import write

TARGET_SAMPLE_RATE = 48000
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
            recording = sd.rec(int(duration * TARGET_SAMPLE_RATE), samplerate=TARGET_SAMPLE_RATE, channels=NUM_CHANNELS)
            num_channels = NUM_CHANNELS
        except sounddevice.PortAudioError:  # Microphone couldn't record in 2-channel mode
            try:
                recording = sd.rec(int(duration * TARGET_SAMPLE_RATE), samplerate=TARGET_SAMPLE_RATE, channels=1)

                num_channels = 1
            except sounddevice.PortAudioError:
                print('Could not record from audio device')
                return None, None

        return recording, num_channels

    def wait(self):
        sd.wait()

    @staticmethod
    def get_devices() -> list[str]:
        devices = sd.query_devices()
        names = list()

        for device in devices:
            names.append(device['name'])

        return names

    @staticmethod
    def preprocess(audio: AudioSegment, num_channels, sample_rate):
        output_path = os.path.join(SoundRecorder.__eng_temp_dir, 'processed_audio.wav')
        if os.path.exists(output_path):
            os.remove(output_path)

        if sample_rate != TARGET_SAMPLE_RATE:
            audio = SoundRecorder.__resample(audio)

        if num_channels != NUM_CHANNELS:
            temp_path = os.path.join(SoundRecorder.__eng_temp_dir, 'temp.wav')
            audio.export(temp_path, format='wav')
            left = AudioSegment.from_wav(temp_path)
            right = AudioSegment.from_wav(temp_path)
            audio = AudioSegment.from_mono_audiosegments(left, right)
            os.remove(temp_path)

        # Write the processed audio to a file
        audio.export(output_path, format='wav')

        return output_path

    @staticmethod
    def __resample(audio: AudioSegment):
        return audio.set_frame_rate(TARGET_SAMPLE_RATE)
