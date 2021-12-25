import os
import numpy as np

from tqdm import tqdm
from pydub import AudioSegment
from scipy.io.wavfile import write


SAMPLE_RATE = 16384


def preprocess_dataset(sound_files, destination_folder, format="mp3"):
    """
    Standardise audio for training

    Preprocess audio files to 1 second wav files resampled at 16k.

    Parameters
    ----------
    sound_files : List[str]
        List of sound file paths
    destination_folder : str
        Desination folder where to save post-processed audio
    format : str
        Type of the input audio (mp3, wav)
    """
    os.makedirs(destination_folder, exist_ok=True)

    for i, path in enumerate(tqdm(sound_files)):
        sound = AudioSegment.from_file(path, format=format)
        resampled = sound.set_frame_rate(SAMPLE_RATE)
        waveform = np.array(resampled.get_array_of_samples())
        second = peak_second(waveform)
        write(f"{destination_folder}/{i}.wav", SAMPLE_RATE, second)


def peak_second(sound):
    """
    Heuristic to segment 1 second in the sound file.

    Locate the peak signal and crop the audio to a 1 second interval.

    Parameters
    ----------
    sound : np.array
        Audio segment to crop
    """
    peak_location = np.argmax(sound)
    offset = SAMPLE_RATE // 2

    if peak_location < offset:
        peak_location = offset
    elif len(sound) - peak_location < offset:
        peak_location = len(sound) - offset

    return sound[peak_location-offset:peak_location+offset]

