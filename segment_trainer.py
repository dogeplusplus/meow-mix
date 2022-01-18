from pathlib import Path

from load.dataset import build_datasets, DataConfig
from networks.crnn import CRNN, CRNNConfig, CRNNTrainConfig


def main():
    dataset_path = Path("data", "catswhoyell")
    data_config = DataConfig(dataset_path, 16, 0.2)
    train_loader, val_loader = build_datasets(data_config)


    model_config = CRNNConfig()
    train_config = CRNNTrainConfig(100, 1e-3, 5)
    model = CRNN(model_config)
    model.train(train_config, train_loader, val_loader)


if __name__ == "__main__":
    main()
