import torch
import numpy as np

from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split


def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Batch uneven audio files using tiling.

    Parameters
    ----------
    batch : List[Tuple[np.ndarray, np.ndarray]]
        List of uneven audio files and labels

    Returns
    -------
    audio_batch: np.ndarray
        Uniformly batched audio signals
    label_batch: np.ndarray
        Uniformly batched audio labels
    """
    audio, labels = zip(*batch)

    max_size = max([l.size for l in labels])
    repeat_fn = lambda x: int(np.ceil(max_size / x.size))
    repetitions = np.vectorize(repeat_fn, otypes=["object"])(labels).astype(int)

    reshaped_audio = [np.tile(a, (1, r))[:, :max_size] for a, r in zip(audio, repetitions)]
    reshaped_labels = [np.tile(l, r)[:max_size] for l, r in zip(labels, repetitions)]

    audio_batch = torch.FloatTensor(reshaped_audio)
    label_batch = torch.FloatTensor(reshaped_labels)

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


@dataclass
class DataConfig:
    directory: Path
    batch_size: int
    val_ratio: float


def build_datasets(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation data loaders for training.

    Parameters
    ----------
    config: DataConfig
        Data configuration such as batch size and validation set.

    Returns
    -------
    train_loader: DataLoader
        Training dataset
    val_loader: DataLoader
        Validation dataset

    Raises
    ------
    AssertionError
        If the validation set ratio is not between 0 and 1
    """
    assert 0 <= config.val_ratio <= 1
    ds = MeowDataset(config.directory)
    val_size = int(len(ds) * config.val_ratio)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=config.batch_size)
    val_loader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=config.batch_size)

    return train_loader, val_loader
