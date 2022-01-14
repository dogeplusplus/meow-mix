from pathlib import Path

from load.dataset import build_datasets, DataConfig
from networks.convolutional import ConvSegment, ConvConfig, ConvTrainConfig


def main():
    dataset_path = Path("data", "preprocessed", "catswhoyell")
    data_config = DataConfig(dataset_path, 16, 0.2)
    train_loader, val_loader = build_datasets(data_config)

    model_config = ConvConfig()
    train_config = ConvTrainConfig(100,1e-4,5)

    model = ConvSegment(model_config)
    model.train(train_config, train_loader, val_loader)


if __name__ == "__main__":
    main()
