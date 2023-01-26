"""
run from 1 directory above sound_tools/
> python -m sound_tools.test_spectrograms
"""
import numpy as np
import logging
from .sound import Sound
from .spectrograms import get_power_spectrum
import matplotlib.pyplot as plt


def test_spectrogram():
    """
    Generate spectrum of random data, make sure different chunk sizes for the piece-wise SFTF are identical.
    """
    data = np.random.randn(100000)
    max_fft_sizes = [10000, 20000, 60000, 100000, 200000]

    results = [get_power_spectrum(data, 44100, resolution_hz=120., max_stft_size=mfs)[0] for mfs in max_fft_sizes]
    for i, result in enumerate(results[:-1]):
        logging.info("Comparing result:  MAX_FFT_SIZE: %i to %i" % (max_fft_sizes[i], max_fft_sizes[i + 1]))
        assert (np.max(np.abs(result - results[i + 1])) < 1e-10), "Results with MAX_FFT_SIZE = %i and %i differ." % (
            max_fft_sizes[i], max_fft_sizes[i + 1])


def _debug_sandbox():
    logging.basicConfig(level=logging.INFO)
    sound = Sound("Aphelocoma_californica_-_California_Scrub_Jay_-_XC110976.wav")

    z, f, t = get_power_spectrum(sound.get_mono_data(), sound.metadata.framerate)

    plt.imshow(np.log(1 + np.abs(z)), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    plt.colorbar()
    plt.show()


def _compare_params():
    """
    Show spectrogram at several different resolutions
    :return:
    :rtype:
    """
    sound = Sound("Aphelocoma_californica_-_California_Scrub_Jay_-_XC110976.wav")
    resolutions = [100., 128., 170.]
    time_res = [0.001, 0.0005, 0.00025]

    ax = None
    for h_i, h_res in enumerate(resolutions):
        for t_i, t_res in enumerate(time_res):
            print(h_res, t_res)
            ax = plt.subplot(len(resolutions), len(time_res), h_i * len(time_res) + t_i + 1) if ax is None else \
                plt.subplot(len(resolutions), len(time_res), h_i * len(time_res) + t_i + 1, sharex=ax, sharey=ax)
            z, f, t = get_power_spectrum(data=sound.get_mono_data(), frame_rate=sound.metadata.framerate,
                                         resolution_hz=h_res, resolution_sec=t_res)
            power = np.abs(z)
            print(z.shape)

            plt.imshow(np.log(4 + power), extent=[t[0], t[-1], f[0], f[-1]], aspect='auto', origin='lower')
            plt.title("dt=%.6f, df = %f" % (t_res, h_res), fontsize=6)
            plt.xlim(0, 1)
            plt.ylim(1000, 9000)
            plt.gca().xaxis.set_visible(False)
            plt.gca().yaxis.set_visible(False)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # _debug_sandbox()
    test_spectrogram()
    print("All tests pass.")
