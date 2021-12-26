import librosa
import logging
import pandas as pd

from pathlib import Path
from librosa.feature import melspectrogram
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class MeowDataset(Dataset):
    def __init__(self, audio_directory):
        self.audio_directory = audio_directory
        label_path = Path(self.audio_directory, "labels.csv")
        self.labels = pd.read_csv(label_path)
        self.audio_paths = list(Path(self.audio_directory).glob("**/*.wav"))

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        waveform, sample_rate = librosa.load(path, sr=None)
        mfcc_sequence = melspectrogram(waveform, sr=sample_rate)
        audio_duration = waveform.size / sample_rate
        return mfcc_sequence

    def __len__(self):
        return len(self.audio_paths)

