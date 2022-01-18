import tqdm
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from typing import Tuple, List
from more_itertools import pairwise
from torch.utils.data import DataLoader
from dataclasses import dataclass, field


@dataclass
class CRNNTrainConfig:
    epochs: int
    learning_rate: float
    save_every: int


@dataclass
class CRNNConfig:
    filters: List = field(default_factory=lambda: [128, 128, 128])
    rnn_hidden_size: int = 32
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (1, 1)
    pooling: List = field(default_factory=lambda: [(5, 1), (2, 1), (2, 1)])
    sequence_length: int = 160


class CRNN(nn.Module):
    NUM_CLASSES = 3

    def __init__(self, model_config: CRNNConfig):
        super().__init__()

        self.convolutional = nn.Sequential()
        self._filters = [1] + model_config.filters
        self._rnn_hidden_size = model_config.rnn_hidden_size
        self._sequence_length = model_config.sequence_length
        for idx, (in_filter, out_filter) in enumerate(pairwise(self._filters)):
            self.convolutional.add_module(f"conv_{idx}",
                nn.Conv2d(
                    in_filter,
                    out_filter,
                    model_config.kernel_size,
                    model_config.stride,
                    model_config.padding
                )
            )
            self.convolutional.add_module(f"relu_{idx}", nn.LeakyReLU(0.01))
            self.convolutional.add_module(f"max_pool_{idx}",
                nn.MaxPool2d(model_config.pooling[idx])
            )

        self.rnn1 = nn.GRU(
            2 * self._filters[-1],
            self._rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn2 = nn.GRU(
            2 * self._rnn_hidden_size,
            self._rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.time_distributed_dense = nn.Conv2d(1, self.NUM_CLASSES, (64, 1))
        self.softmax = nn.Softmax(dim=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accumulation_steps = 16
        self.loss_fn = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.unsqueeze(1)
        x = self.convolutional(x)
        x = x.view(batch_size, -1, 2 * self._filters[-1])
        x, h = self.rnn1(x)
        x, _ = self.rnn2(x, h)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.time_distributed_dense(x)
        x = self.softmax(x)
        x = torch.squeeze(x)
        x = x.view(batch_size, -1, self.NUM_CLASSES)

        return x


    def train(self, config: CRNNTrainConfig, train_dataset: DataLoader, val_dataset: DataLoader = None):
        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        self.optimiser = optim.Adam(self.parameters())
        self.optimiser.zero_grad()
        for e in range(config.epochs):
            train_losses = {
                "loss": 0,
                "accuracy": 0,
            }
            val_losses = deepcopy(train_losses)

            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                total=len(train_dataset),
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for i, batch in enumerate(train_bar):
                losses = self.train_step(i, batch)
                for loss in train_losses:
                    train_losses[loss] = (train_losses[loss] * i + losses[loss]) / (i+1)
                self.display_metrics(train_losses, train_bar)

            for name, value in train_losses.items():
                mlflow.log_metric(f"train_{name}", float(value), step=e)

            if val_dataset is not None:
                val_bar = tqdm.tqdm(
                    val_dataset,
                    desc=val_desc.format(e+1),
                    total=len(val_dataset),
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                )
                for i, batch in enumerate(val_bar):
                    losses = self.val_step(batch)
                    for loss in val_losses:
                        val_losses[loss] = (val_losses[loss] * i + losses[loss]) / (i+1)
                    self.display_metrics(val_losses, val_bar)

                for name, value in val_losses.items():
                    mlflow.log_metric(f"val_{name}", float(value), step=e)

            if (e + 1) % config.save_every == 0:
                mlflow.pytorch.log_model(self, "model")

        mlflow.pytorch.log_model(self, "model")

    def train_step(self, idx, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        pred = self(x)
        loss = self.loss_fn(pred, y.float())
        classification = torch.argmax(pred, dim=-1)
        ground_truth = torch.argmax(y, dim=-1)

        accuracy = (classification == ground_truth).sum() / torch.numel(ground_truth)
        loss.backward()
        if (idx + 1) % self.accumulation_steps == 0:
            self.optimiser.step()
            self.optimiser.zero_grad()

        return {
            "loss": loss.cpu().detach().numpy(),
            "accuracy": accuracy.cpu().detach().numpy(),
        }

    def val_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            pred = self(x)
            loss = self.loss_fn(pred, y)
            classification = torch.argmax(pred, dim=-1)
            ground_truth = torch.argmax(y, dim=-1)
            accuracy = (classification == ground_truth).sum() / torch.numel(ground_truth)

        return {
            "loss": loss.cpu().detach().numpy(),
            "accuracy": accuracy.cpu().detach().numpy(),
        }

    def display_metrics(self, metrics, progress_bar):
        evaluated_metrics = {
                k: f"{v:.5f}"
            for k, v in metrics.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)
