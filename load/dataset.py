import glob
import librosa
import torchaudio
import pandas as pd

from librosa.feature import mfcc
from torch.utils.data import Dataset


class MeowDataset(Dataset):
    def __init__(self, audio_directory):
        self.audio_directory = audio_directory
        label_path = f"{self.audio_directory}/labels.csv"
        self.labels = pd.read_csv(label_path)
        self.audio_paths = glob.glob(f"{self.audio_directory}/*.wav")

    def __getitem__(self, idx):
        waveform, _ = librosa.load(self.audio_paths[idx])
        mfcc_sequence = mfcc(waveform, n_mfcc=40)
        return mfcc_sequence

    def __len__(self):
        return len(self.audio_paths)
