
from pathlib import Path
from torch.utils.data import DataLoader

from load.dataset import MeowDataset, collate_fn
from networks.yellnet import YellNet, SegmentConfig, SegmentTrainConfig

ds = MeowDataset(Path("data", "preprocessed", "catswhoyell"))
loader = DataLoader(ds, collate_fn=collate_fn, batch_size=16)
model_config = SegmentConfig()
train_config = SegmentTrainConfig(100,1e-4)

model = YellNet(model_config)
# model.train(loader, train_config)

import torch
for x, y in loader:
    pred = model(torch.FloatTensor(x))
    import pdb; pdb.set_trace()
