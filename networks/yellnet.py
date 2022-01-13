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
class SegmentTrainConfig:
    epochs: int
    learning_rate: float
    save_every: int


@dataclass
class SegmentConfig:
    filters: List = field(default_factory=lambda: [32, 64, 32, 16])
    kernel_size: Tuple[int, int] = (3, 5)
    stride: Tuple[int, int] = (2, 1)
    padding: Tuple[int, int] = (0, 2)


class YellNet(nn.Module):
    def __init__(self, model_config: SegmentConfig):
        super().__init__()

        self.network = nn.Sequential()
        self._filters = [1] + model_config.filters
        for idx, (in_filter, out_filter) in enumerate(pairwise(self._filters)):
            self.network.add_module(f"conv_{idx}",
                nn.Conv2d(
                    in_filter,
                    out_filter,
                    model_config.kernel_size,
                    model_config.stride,
                    model_config.padding
                )
            )
            self.network.add_module(f"relu_{idx}", nn.LeakyReLU(0.1))
            self.network.add_module(f"dropout_{idx}", nn.Dropout(0.3))


        self.last_conv = nn.Conv2d(self._filters[-1], 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accumulation_steps = 16
        self.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        x = self.last_conv(x)
        x = self.sigmoid(x)

        return x


    def train(self, config: SegmentTrainConfig, train_dataset: DataLoader, val_dataset: DataLoader = None):
        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        self.optimiser = optim.Adam(self.parameters())
        self.optimiser.zero_grad()
        for e in range(config.epochs):
            train_losses = {
                f"loss": 0,
                f"accuracy": 0,
            }
            val_losses = deepcopy(train_losses)

            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                total=len(train_dataset),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
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
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
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

        return self.model

    def train_step(self, idx, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        pred = self(x)
        loss = self.loss_fn(torch.squeeze(pred), y)
        classification = pred > 0.5
        accuracy = (classification == y).sum() / len(y)

        loss.backward()
        if (idx + 1) % self.accumulation_steps == 0:
            self.optimiser.step()
            self.optimiser.zero_grad()

        return {
            f"loss": loss.cpu().detach().numpy(),
            f"accuracy": accuracy.cpu().detach().numpy(),
        }

    def val_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            pred = self(x)
            loss = self.loss_fn(torch.squeeze(pred), y)
            classes = torch.argmax(pred, dim=-1)
            accuracy = (classes == y).sum() / len(y)

        return {
            f"loss": loss.cpu().detach().numpy(),
            f"accuracy": accuracy.cpu().detach().numpy(),
        }

    def display_metrics(self, metrics_dict, progress_bar):
        evaluated_metrics = {
            k: str(v)[:7]
            for k, v in metrics_dict.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)
