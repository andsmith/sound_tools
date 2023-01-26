from sound import SoundPlayer, Sound
import logging
import time
import matplotlib.pyplot as plt
import pyaudio


def sound_test(file_name):
    sound = Sound(file_name)
    data = sound.data[0]
    #print(data.dtype)
    #plt.plot(data[44100 * 4:44100 * 5])
    #plt.show()
    duration = sound.metadata.nframes / float(sound.metadata.framerate)
    logging.info("Read %.2f sec sound." % (duration,))

    position = [0]
    finished = [False]
    buffer_size = 1024

    def _make_samples(n_frames):
        endpoint = position[0] + n_frames
        if endpoint > data.size:
            endpoint = data.size
            logging.info("Sound finished.")
            finished[0] = True
        samples = data[position[0]:endpoint]
        position[0] = endpoint
        bytes = sound.encode_samples(samples)
        return bytes
    player = SoundPlayer.from_sound(sound, _make_samples, frames_per_buffer=buffer_size)
    player.start()
    while not finished[0]:
        time.sleep(.1)

    
    logging.info("... stream finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filename = "Aphelocoma_californica_-_California_Scrub_Jay_XC110976.wav"
    sound_test(filename)
    print("Test complete.")
