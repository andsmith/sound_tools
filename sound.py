import numpy as np
import os
import subprocess
import tempfile
import logging
import shutil
import wave

from collections import namedtuple

# copied from wave.wave.py, not sure how to do this pythonically
_wave_params = namedtuple('_wave_params', 'nchannels sampwidth framerate nframes comptype compname')


class Sound(object):
    """
    Class to hold data from a sound file
    """
    NATIVE_FORMATS = [("Waveform Audio File Format", ('*.wav',))]
    OTHER_FORMATS = [("Windows Media Audio", ('*.m4a',)),
                     ("Ogg Vorbis", ('*.ogg', '*.oga')),
                     ("MPEG-1/2 Audio Layer III", ('*.mp3',)),
                     ('Free Lossless Audio Codec', ("*.flac",))]

    def __init__(self, filename=None, framerate=44100, sampwidth=2, nchannels=1, comptype='NONE',
                 compname='not compressed'):
        if filename is not None:
            self._filename = filename
            self.data, self.metadata, self.data_raw = Sound._read_sound(filename)
            self.duration_sec = (self.metadata.nframes - 1) / float(self.metadata.framerate)
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
        return convert_to_bytes(samples, self.data[0].dtype)

    @staticmethod
    def _read_sound(filename):
        # remove "*" for extension check
        natives = [ext[1:] for fmt in Sound.NATIVE_FORMATS for ext in fmt[1]]
        others = [ext[1:] for fmt in Sound.NATIVE_FORMATS for ext in fmt[1]]
        ext = os.path.splitext(filename)[1].lower()
        if ext in natives:
            return Sound._read_wav(filename)
        elif ext in others:
            return Sound._read_other(filename)
        else:
            raise Exception("unknown file type, not one of %s:  %s" % (natives + others, ext))

    @staticmethod
    def _read_wav(filename):
        with wave.open(filename, 'rb') as wav:
            wav_params = wav.getparams()
            data_raw = wav.readframes(wav_params.nframes)
        data = convert_from_bytes(data_raw, wav_params)
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

    def set_data(self, channel_data):
        """
        :param channel_data:  list of numpy arrays
        """
        self.data = channel_data
        self.data_raw = convert_to_bytes(self.data, get_encoding_type(self.metadata))
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
            new_bytes = convert_to_bytes(data, dtype)
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


def convert_from_bytes(data, wav_params):
    # figure out data type
    n_data = np.frombuffer(data, get_encoding_type(wav_params))
    # separate interleaved channel data
    n_data = [n_data[offset::wav_params.nchannels] for offset in range(wav_params.nchannels)]

    return n_data


def convert_to_bytes(chan_float_data, data_type=None):
    data_type = chan_float_data[0].dtype if data_type is None else data_type
    # interleave channel data
    n_chan = len(chan_float_data)
    data = np.zeros(n_chan * chan_float_data[0].size, dtype=data_type)
    for i_chan in range(n_chan):
        data[i_chan::n_chan] = chan_float_data[i_chan]

    return data.tobytes()


ENCODING_TYPES_FROM_SAMPLE_WIDTH = {1: np.uint8,
                  2: np.int16,
                  4: np.int32}


def get_encoding_type(wav_params):
    """
    Find numpy equivalent for different file formats
    :param wav_params:  metadata
    :return:  numpy dtype
    """
    try:
        return ENCODING_TYPES_FROM_SAMPLE_WIDTH[wav_params.sampwidth]
    except KeyError:
        raise Exception("Don't know data type for sample-width = %i." % (wav_params.sampwidth,))
