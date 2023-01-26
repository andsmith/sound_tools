# sound_tools

class `sound.Sound`:

* Load sound files, decode data into numpy arrays of sound samples.
* Save sound files.
* Draw a simple visualization of the sound to an image.

class `sound.SoundPlayer`:

* Play live sound.

function `spectrograms.get_power_spectrum`:

* For a given time and frequency resolution, run the Short Time Fourier Transform (STFT) to produce a spectrogram, i.e.
  the power of frequency *f* at time *t*.