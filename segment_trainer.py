from pathlib import Path

from load.dataset import build_datasets, DataConfig
from networks.yellnet import YellNet, SegmentConfig, SegmentTrainConfig


def main():
    dataset_path = Path("data", "preprocessed", "catswhoyell")
    data_config = DataConfig(dataset_path, 16, 0.2)
    train_loader, val_loader = build_datasets(data_config)

    model_config = SegmentConfig()
    train_config = SegmentTrainConfig(100,1e-4,5)

    model = YellNet(model_config)
    model.train(train_config, train_loader, val_loader)

if __name__ == "__main__":
    main()
