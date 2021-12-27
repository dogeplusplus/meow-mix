import logging
import numpy as np

from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def collate_fn(batch):
    audio, labels = zip(*batch)

    max_size = max([l.size for l in labels])
    repeat_fn = lambda x: int(np.ceil(max_size / x.size))
    repetitions = np.vectorize(repeat_fn, otypes=["object"])(labels).astype(int)

    reshaped_audio = [np.tile(a, (1, r))[:, :max_size] for a, r in zip(audio, repetitions)]
    reshaped_labels = [np.tile(l, r)[:max_size] for l, r in zip(labels, repetitions)]

    audio_batch = np.array(reshaped_audio)
    label_batch = np.array(reshaped_labels)

    return audio_batch, label_batch


class MeowDataset(Dataset):
    def __init__(self, data_dir):
        audio_dir = data_dir.joinpath("audio")
        label_dir = data_dir.joinpath("labels")

        self.audio_paths = list(audio_dir.glob("**/*.npy"))
        self.label_paths = list(label_dir.glob("**/*.npy"))

    def __getitem__(self, idx):
        wav_path = self.audio_paths[idx]
        label_path = self.label_paths[idx]
        audio = np.load(wav_path)
        label = np.load(label_path)
        return audio, label

    def __len__(self):
        return len(self.audio_paths)

