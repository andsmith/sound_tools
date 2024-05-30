import logging
import pyaudio
import numpy as np
from .pcm_data import DEFAULT_ENCODING_FOR_SAMPLE_WIDTH, convert_to_bytes, NUMPY_TYPES_FOR_PCM_DATA_TYPE


def get_dtype_range(data_type):
    return np.iinfo(data_type).min, np.iinfo(data_type).max,


class Encoder(object):
    """
    Encode samples normalized to [-1, 1].
    """

    def __init__(self, sample_width):
        #import ipdb; ipdb.set_trace()
        self._encoding = DEFAULT_ENCODING_FOR_SAMPLE_WIDTH[sample_width]
        self._dtype = NUMPY_TYPES_FOR_PCM_DATA_TYPE[self._encoding]
        self._sample_range = get_dtype_range(self._dtype)
        self._sample_span = self._sample_range[1] - self._sample_range[0]

    def encode(self, channel_data):
        """
        floats in [-1, 1] -> dtypes in [dtype min, dtype max]
        """
        samples = [((1 + chan) / 2 * self._sample_span + self._sample_range[0]).astype(self._dtype)
                   for chan in channel_data]
        return convert_to_bytes(samples, encoding=self._encoding)


class SoundPlayer(object):
    def __init__(self, sample_width, frame_rate, channels, sample_generator, frames_per_buffer=None):
        """
        Open stream for playing.
        :param sample_width:  bytes per frame
        :param frame_rate:  Samples per second
        :param channels: 1 or 2
        :param sample_generator:  Function taking 1 int (n_samples) and returning that many samples
        """
        self._sample_width = sample_width
        self._frame_rate = frame_rate
        self._channels = channels
        self._sample_gen = sample_generator
        self._p = pyaudio.PyAudio()
        self._buffer_size = frames_per_buffer if frames_per_buffer is not None else 1024 * 2
        self._stream = None

    @staticmethod
    def from_sound(sound, sample_generator, frames_per_buffer=None, **kwargs):
        """
        Init player with params loaded from file.
        :param sound: Sound object
        :param sample_generator:  see __init__
        :param frames_per_buffer:  see __init
        :return: SoundPlayer
        """
        return SoundPlayer(sample_width=sound.metadata.sampwidth,
                           frame_rate=sound.metadata.framerate,
                           channels=sound.metadata.nchannels,
                           sample_generator=sample_generator, frames_per_buffer=frames_per_buffer)

    def start(self):
        logging.info("Starting playback...")
        self._stream = self._p.open(format=self._p.get_format_from_width(self._sample_width),
                                    channels=self._channels,
                                    rate=self._frame_rate,
                                    output=True,
                                    frames_per_buffer=self._buffer_size,
                                    stream_callback=self._get_samples)

    def _get_samples(self, in_data, frame_count, time_info, status):
        """
        (Pyaudio callback)
        pyaudio wants more samples, so get them from the callback
        :param in_data: pyaudio param
        :param frame_count:
        :param time_info: pyaudio param
        :param status: pyaudio param
        :return:  frame_count samples, or fewer if at the end of the sound, and the appropriate code
        """
        data = self._sample_gen(frame_count)

        code = pyaudio.paContinue
        if len(data) < frame_count:
            self._stream = False
            code = pyaudio.paComplete

        return data, code

    def stop(self):
        logging.info("Playback stopped.")
        self._stream.close()
        self._stream = None

    def shutdown(self):
        self._p.terminate()
