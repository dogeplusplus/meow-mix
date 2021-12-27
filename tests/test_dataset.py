import numpy as np

from load.dataset import collate_fn

def test_collate_fn():
    audio_samples = [
        np.zeros((40, 100)),
        np.tile(np.arange(10)[np.newaxis, :], (40, 1))
    ]


    label_samples = [
        np.zeros(100),
        np.arange(10),
    ]

    batch = zip(audio_samples, label_samples)

    actual_audio, actual_label = collate_fn(batch)

    assert actual_audio.shape == (2, 40, 100)
    assert actual_label.shape == (2, 100)

    assert np.array_equal(actual_label[1], np.tile(np.arange(10), 10))
