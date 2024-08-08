from scipy.signal import medfilt, stft, spectrogram
import numpy as np
import logging
from .sound import Sound


def get_stft_params():
    return dict(
        fs=None,  # frame_rate,
        nperseg=None,  # window_size,
        noverlap=None,  # overlap,
        padded=True,
        boundary=None)


def get_power_spectrum(data, frame_rate, resolution_hz=110.0, resolution_sec=0.0005, freq_range=None,
                       max_stft_size=25000):
    """
    Get a short-time fft (i.e. with a sliding window) of a signal.

    Higher values of resolution_hz resolve higher frequencies better at the expense of lower frequencies.
    Higher values of resolution_sec are necessary to resolve higher frequencies, but will blur out the lower. etc.
    (Do more passes with different values to resolve different areas of the spectrum, etc.)

    (Default params are somewhat optimized for visualizing birdsong spectra.)

    :param data:  mono signal (n frames)
    :param frame_rate:  frames per sec
    :param resolution_hz:  Spectrum will bin frequencies in bins of this size.
    :param resolution_sec:  Window will slide along in steps of this duration.
    :param freq_range:  return frequencies in this band (float low, float high), None = (0, 2/frame_rate)
    :param max_stft_size:  If data.size > max_stft_size, do calculation piecewise (avoid memory blow-up)

    :return: complex - f x t array of z values (fft-result), where f and t are:
             floats - array of f frequency bins (bin centers, disregarding DC component)
             floats - array of t window times (window centers)
    """
    frame_rate = float(frame_rate)

    # STFT params
    params = get_stft_params()
    step_size = int(resolution_sec * frame_rate)
    window_size = int(frame_rate / resolution_hz)
    window_size += (window_size % 2)  # make even
    params['noverlap'] = window_size - step_size
    params['nperseg'] = window_size
    params['fs'] = frame_rate

    # Block-wise param, dispose of invalid data
    padding_samples = window_size // 2

    def prune_frequencies(freq, z_vals):
        f_range = np.sum(freq < freq_range[0]), np.sum(freq < freq_range[1])
        return freq[f_range[0]:f_range[1]], z_vals[f_range[0]:f_range[1], :]

    if padding_samples * 3 > data.size:
        f, t, z = stft(data, **params)
        f, z = prune_frequencies(f, z)
        return z, f, t

    padding_duration_sec = padding_samples / frame_rate
    freq_range = freq_range if freq_range is not None else (
        0., 2. / frame_rate)

    # make sure chunk_size has whole number of windows
    chunk_size = max_stft_size
    remainder = chunk_size - window_size
    chunk_size -= remainder % params['noverlap']

    logging.info("Calculating power spectrum for %i samples, %i-wide windows, at %i-sample intervals" %
                 (data.size, window_size, step_size))

    # calculate the timestamp of each column of the power spectrum
    window_center_inds = np.arange(
        # indices
        padding_samples, data.size, window_size - params['noverlap'])
    window_center_inds = window_center_inds[window_center_inds <=
                                            data.size - padding_samples]
    window_center_times = window_center_inds / frame_rate

    def get_chunk_spectrum(w_start, w_end):  # index into window_center_*
        """
        Get a (time) section of the spectrogram.  Remove unwanted frequencies.
        indeed by time-slice (bin)
        :param w_start:  into window_center_inds
        :param w_end:  into window_center_inds
        :return: array of f floats - frequency bin (centers)
                 array of t floats - bin centers, as returned by scipy.fft
                 f x t array of complex - the z values
        """

        data_range = window_center_inds[w_start] - \
            padding_samples, window_center_inds[w_end] + padding_samples
        f, t, z = stft(data[data_range[0]:
                            data_range[1]],
                       **params)
        assert (t.size - 1 == (w_end - w_start))

        f, z = prune_frequencies(f, z)
        return f, t, z

    n_windows_per_chunk = int(np.floor(chunk_size - window_size) / step_size)

    z_vals = []
    times = []
    next_w_ind = 0

    while True:
        if next_w_ind + n_windows_per_chunk > window_center_inds.size:
            end_index = window_center_inds.size - 1
        else:
            end_index = next_w_ind + n_windows_per_chunk
        fr, tm, zv = get_chunk_spectrum(next_w_ind, end_index)
        freqs = fr  # doesn't change

        times.append(tm - padding_duration_sec +
                     window_center_times[next_w_ind])
        z_vals.append(zv)
        next_w_ind += tm.size

        if next_w_ind >= window_center_inds.size:
            break

    frequencies = freqs
    z_values = np.hstack(z_vals)
    timestamps = np.hstack(times)
    logging.info("\t...complete with %i frequency bins and %i time windows." % (
        frequencies.size, timestamps.size))
    return z_values, frequencies, timestamps, params
