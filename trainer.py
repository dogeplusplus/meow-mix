from torch.utils.data import DataLoader

from networks.wavegan import WaveGan, TrainingConfig, ModelConfig
from load.dataset import MeowDataset

def main():
    batch_size = 64
    learning_rate = 1e-3
    epochs = 201
    batch_size = 2

    paths = "data/meow_dataset"
    ds = MeowDataset(paths)

    model_config = ModelConfig()
    train_config = TrainingConfig(model_config)
    model = WaveGan(train_config)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    model.train(train_loader)



if __name__ == "__main__":
    main()

