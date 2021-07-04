import os
import torch
import torchaudio
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt

from random import shuffle
from torchaudio.transforms import MelSpectrogram

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)

    figure.suptitle(title)
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(lim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_multiple_spectograms(waveforms, sample_rates):
    num_waves = len(waveforms)
    width = int(np.sqrt(num_waves))

    fig, ax = plt.subplots(width, width)
    for i in range(width):
        for j in range(width):
            index = i*width + j
            ax[i,j].specgram(waveforms[index], Fs=sample_rates[index])
            # ax[i,j].axis("off")

    plt.show(block=False)


def play_sound(sound_path):
    wave_obj = sa.WaveObject.from_wave_file(sound_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def main():
    WAV_DIR = "data/dataset"
    sound_files = os.listdir(WAV_DIR)
    index = 8
    sound_path = os.path.join(WAV_DIR, sound_files[index])

    # waveform, sample_rate = torchaudio.load(sound_path)
    # plot_waveform(waveform, sample_rate)
    # plot_specgram(waveform, sample_rate)

    waveforms = []
    sample_rates = []
    shuffle(sound_files)

    for i in range(4):
        sound_path = os.path.join(WAV_DIR, sound_files[i])
        waveform, sample_rate = torchaudio.load(sound_path)
        waveforms.append(waveform.numpy())
        sample_rates.append(sample_rate)

    plot_multiple_spectograms(waveforms, sample_rates)


if __name__ == "__main__":
   main()

