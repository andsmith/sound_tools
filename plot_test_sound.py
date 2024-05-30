from .sound import Sound
import logging
import os
import matplotlib.pyplot as plt


def run_test():
    s = Sound(os.path.join(os.path.split(__file__)[0], "test_sound_s24le.wav"))
    x = s.get_mono_data()
    plt.plot(x)
    plt.show()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    run_test()
