import sounddevice as sd
import librosa
from pydub import AudioSegment
from scipy.io.wavfile import write

SAMPLE_RATE = 48000
NUM_CHANNELS = 2


class SoundRecorder:

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
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        return recording

    @staticmethod
    def get_devices() -> list[str]:
        devices = sd.query_devices()
        names = list()

        for device in devices:
            names.append(device['name'])

        return names

    @staticmethod
    def preprocess(audio, num_channels, sample_rate):
        data_type = 'ndarray'

        if sample_rate != SAMPLE_RATE:
            audio = SoundRecorder.__resample(audio, sample_rate)

        if num_channels != NUM_CHANNELS:
            write('eng_temp/temp.wav', SAMPLE_RATE, audio)
            left = AudioSegment.from_wav('eng_temp/temp.wav')
            right = AudioSegment.from_wav('eng_temp/temp.wav')
            audio = AudioSegment.from_mono_audiosegments(left, right)
            data_type = 'audiosegment'

        # Write the processed audio to a file
        if data_type == 'ndarray':
            write('eng_temp/output.wav', SAMPLE_RATE, audio)
        else:
            audio.export('eng_temp/output.wav', format='wav')

    @staticmethod
    def __resample(audio, orig_sample_rate):
        return librosa.resample(audio, orig_sr=orig_sample_rate, target_sr=SAMPLE_RATE)
