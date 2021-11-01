from torch.utils.data import DataLoader

from networks.wavegan import WaveGan
from data_loader.dataset import MeowDataset

def main():
    batch_size = 64
    learning_rate = 1e-3
    epochs = 100
    batch_size = 2

    d = 8
    c = 1
    model = WaveGan(c, d)
    paths = "data/meow_dataset"
    ds = MeowDataset(paths)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    model.train(epochs, train_loader)


if __name__ == "__main__":
    main()

