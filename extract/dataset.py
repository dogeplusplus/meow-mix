import glob
import torchaudio
from torch.utils.data import Dataset



class MeowDataset(Dataset):
    def __init__(self, audio_directory):
        self.audio_directory = audio_directory

        self.audio_paths = glob.glob(f"{self.audio_directory}/*.wav")

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.audio_paths[idx])
        return waveform

    def __len__(self):
        return len(self.audio_paths)

