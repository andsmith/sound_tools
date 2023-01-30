"""
Generate sound samples using fm synthesis
See http://www.cs.cmu.edu/~music/icm-online/readings/fm-synthesis/index.html

"""
import numpy as np


class FMSynthesizer(object):
    """
    Use as input to SoundPlayer
    """

    def __init__(self, rate=44100.0, carrier_init=(440.0, 0.6), modulation_init=(1.0, 0.0)):
        self._rate = rate
        self._params = {'carrier_freq': carrier_init[0],
                        'carrier_amp': carrier_init[1],
                        'mod_freq': modulation_init[0],
                        'mod_depth': modulation_init[1]}
        self._new_params = self._params.copy()
        self._carrier_phase, self._mod_phase = 0.0, 0.0

    def __str__(self):
        return "FMSynth(C_f = %.3f Hz, C_a = %.2f %%, M_f = %.3f Hz. M_d = %.2f)" % (self._params['carrier_freq'],
                                                                                     self._params['carrier_amp'] * 100,
                                                                                     self._params['mod_freq'],
                                                                                     self._params['mod_depth'])

    def set_param(self, name, value):
        """
        Remember the previous value for smooth interpolation
        TODO:   Keep ALL previous values, tag w/timestamp, so values changing too fast aren't lost.
        """
        self._new_params[name] = value

    def reset(self):
        self._params = self._new_params.copy()
        self._carrier_phase, self._mod_phase = 0.0, 0.0

    def get_samples(self, n, advance=True, encode_func=None):
        """
        Get the next N samples and the time of sample n+1
        returns floats in [-1,1], or bytes
        """
        mod_phase_increments = np.cumsum(
            np.hstack([0, np.linspace(2. * np.pi * self._params['mod_freq'] / self._rate,
                                      2. * np.pi * self._new_params['mod_freq'] / self._rate, n)]))
        car_phase_increments = np.cumsum(
            np.hstack([0, np.linspace(2. * np.pi * self._params['carrier_freq'] / self._rate,
                                      2. * np.pi * self._new_params['carrier_freq'] / self._rate, n)]))

        depth = np.linspace(self._params['mod_depth'], self._new_params['mod_depth'], n)
        amp = np.linspace(self._params['carrier_amp'], self._new_params['carrier_amp'], n)

        modulation = np.sin(self._mod_phase + mod_phase_increments[:-1]) * depth
        samples = np.sin(self._carrier_phase + car_phase_increments[:-1] + modulation) * amp

        if advance:
            self._params = self._new_params.copy()
            self._carrier_phase += car_phase_increments[-1]  # np.mod(car_phase_increments[-1], np.pi * 2.0)
            self._mod_phase += mod_phase_increments[-1]  # np.mod(mod_phase_increments[-1], np.pi * 2.0)

        if encode_func is not None:
            return encode_func(samples)

        return samples
