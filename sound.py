import numpy as np
import os
import subprocess
import tempfile
import logging
import shutil
import wave
import pyaudio

from collections import namedtuple

# copied from wave.wave.py, not sure how to do this pythonically
_wave_params = namedtuple('_wave_params', 'nchannels sampwidth framerate nframes comptype compname')


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
    def from_sound(sound, sample_generator, frames_per_buffer=None):
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
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.

        code = pyaudio.paContinue
        if len(data) < frame_count:
            self._stream = False
            code = pyaudio.paComplete

        return data, code

    def stop(self):
        self._stream.close()
        self._stream = None

    def shutdown(self):
        self._p.terminate()


class Sound(object):
    """
    Class to hold data from a sound file
    """
    EXTENSIONS = ['.m4a', '.ogg', '.mp3', '.oga']

    def __init__(self, filename=None, framerate=44100, sampwidth=2, nchannels=1, comptype='NONE',
                 compname='not compressed'):
        if filename is not None:
            self._filename = filename
            self.data, self.metadata, self.data_raw = Sound._read_sound(filename)
            self.duration_sec = (self.metadata.nframes-1) / float(self.metadata.framerate)
        else:
            self._filename = None
            self.metadata = _wave_params(framerate=framerate, sampwidth=sampwidth, comptype=comptype,
                                         compname=compname, nchannels=nchannels, nframes=0)
            self.data = np.array([], dtype=get_encoding_type(self.metadata))
            self.data_raw = bytes([])
            self._duration_sec = 0.

    def get_mono_data(self):
        """
        Avg of all channels.
        :return: numpy array
        """
        if len(self.data) == 1:
            return self.data[0]
        return np.mean(self.data, axis=0)

    def encode_samples(self, samples):
        return Sound._convert_to_bytes(samples, self.data[0].dtype)

    @staticmethod
    def _read_sound(filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.wav':
            return Sound._read_wav(filename)
        elif ext in Sound.EXTENSIONS:
            return Sound._read_other(filename)
        else:
            raise Exception("unknown file type, not one of %s:  %s" % (Sound.EXTENSIONS, ext))

    @staticmethod
    def _read_wav(filename):
        with wave.open(filename, 'rb') as wav:
            wav_params = wav.getparams()
            data_raw = wav.readframes(wav_params.nframes)
        data = Sound._convert_from_bytes(data_raw, wav_params)
        duration = wav_params.nframes / float(wav_params.framerate)
        logging.info("Read file:  %s (%.4f sec, %i Hz, %i channel(s))" % (filename, duration,
                                                                          wav_params.framerate,
                                                                          wav_params.nchannels))

        return data, wav_params, data_raw

    @staticmethod
    def _read_other(filename):
        temp_dir = tempfile.mkdtemp()
        in_stem = os.path.split(os.path.splitext(filename)[0])[1]
        temp_wav = os.path.join(temp_dir, "%s.wav" % (in_stem,))
        logging.info("Converting:  %s  -->  %s" % (filename, temp_wav))
        cmd = ['ffmpeg', '-i', filename, temp_wav]
        logging.info("Running:  %s" % (" ".join(cmd)))
        _ = subprocess.run(cmd, capture_output=True)
        sound = Sound._read_wav(temp_wav)
        shutil.rmtree(temp_dir)
        return sound

    @staticmethod
    def _convert_from_bytes(data, wav_params):
        # figure out data type
        n_data = np.frombuffer(data, get_encoding_type(wav_params))
        # separate interleaved channel data
        n_data = [n_data[offset::wav_params.nchannels] for offset in range(wav_params.nchannels)]

        return n_data

    @staticmethod
    def _convert_to_bytes(chan_float_data, data_type):
        # interleave channel data
        n_chan = len(chan_float_data)
        data = np.zeros(n_chan * chan_float_data[0].size, dtype=data_type)
        for i_chan in range(n_chan):
            data[i_chan::n_chan] = chan_float_data[i_chan]

        return data.tobytes()

    def set_data(self, channel_data):
        """
        :param channel_data:  list of numpy arrays
        """
        self.data = channel_data
        self.data_raw = Sound._convert_to_bytes(self.data, get_encoding_type(self.metadata))
        self.metadata = self.metadata._replace(nframes=channel_data[0].size)
        self.duration_sec = float(self.metadata.nframes) / self.metadata.framerate

    def write(self, filename):
        return self.write_data(filename, data_raw=self.data_raw)

    def write_data(self, filename, data=None, data_raw=None):
        """
        Create a sound file with given data, using same params as self.
        :param filename:  to save as
        :param data:  list of channel data (numpy arrays of samples), or None if using 'data_raw'
        :param data_raw:  bytes() array, or None if using 'data'
        :return:  filename written
        """
        if data is not None and type(data) is not list and not all([type(chan_data) == np.array for chan_data in data]):
            raise Exception("data must be list of numpy arrays")
        if data_raw is not None and type(data_raw) is not bytes:
            raise Exception("data_raw must be bytes array")

        if data is not None:
            dtype = self.data[0].dtype
            new_bytes = self._convert_to_bytes(data, dtype)
        else:
            new_bytes = data_raw

        n_frames = int(len(new_bytes) / self.metadata.sampwidth)
        new_params = self.metadata._replace(nframes=n_frames)
        logging.info("Writing file:  %s" % (filename,))
        with wave.open(filename, 'wb') as wav:
            wav.setparams(new_params)
            wav.writeframesraw(new_bytes)
        duration = new_params.nframes / float(self.metadata.framerate)
        logging.info("\tWrote %.4f seconds of audio data (%i samples)." % (duration, new_params.nframes))
        return filename

    def draw_waveform(self, image, bbox=None, color=(255, 255, 255, 255)):
        """
        Draw waveform on an image.
        :param image:  draw on this image
        :param bbox:  dict with 'top', 'bottom','left','right', bounds within image to draw (scaled to max amplitude)
        :param color: draw waveform  in this color
        """
        if bbox is None:
            bbox = {'top': 0, 'bottom': image.shape[0], 'left': 0, 'right': image.shape[1]}

        data = self.get_mono_data()
        audio_mean = np.mean(data)
        # bin audio into number of horizontal pixels, get max & min for each one
        width = bbox['right'] - bbox['left']
        bin_size = int(data.size / width)

        partitions = data[:bin_size * width].reshape(width, bin_size)
        max_vals, min_vals = np.max(partitions - audio_mean, axis=1), np.min(partitions - audio_mean, axis=1)
        audio_max, audio_min = np.max(max_vals), np.min(min_vals)

        y_center = int((bbox['bottom'] + bbox['top']) / 2)
        y_height = int((bbox['bottom'] - bbox['top']) / 2) * .95
        y_values_high = y_center + np.int64(max_vals / audio_max * y_height)
        y_values_low = y_center - np.int64(min_vals / audio_min * y_height)

        for x in range(bbox['left'], bbox['right']):
            image[y_values_low[x]:y_values_high[x] - 1, x, :] = color


def get_encoding_type(wav_params):
    """
    Find numpy equivalent for different file formats
    :param wav_params:  metadata
    :return:  numpy dtype
    """
    try:
        return {1: np.uint8,
                2: np.int16,
                4: np.int32}[wav_params.sampwidth]
    except KeyError:
        raise Exception("Don't know data type for sample-width = %i." % (wav_params.sampwidth,))
