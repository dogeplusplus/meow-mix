from torch.utils.data import DataLoader

from load.dataset import MeowDataset
from networks.wavegan import WaveGan, GanTrainConfig, GanConfig

def main():
    batch_size = 64

    paths = "data/meow_dataset"
    ds = MeowDataset(paths)

    model_config = GanConfig()
    train_config = GanTrainConfig(model_config)
    model = WaveGan(train_config)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    model.train(train_loader)


if __name__ == "__main__":
    main()

