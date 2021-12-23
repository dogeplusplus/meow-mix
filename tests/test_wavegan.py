import torch
import pytest
import numpy as np

from networks.wavegan import WaveGan, PhaseShuffle, Generator, Discriminator

torch.manual_seed(0)


@pytest.fixture
def model():
    return WaveGan(1, 4)


def test_discriminator_output():
    c = 1
    d = 8
    n = 2
    disc = Discriminator(c, d)

    sound_batch  = torch.ones((n, c, 16384))
    predicted = disc(sound_batch)
    assert predicted.shape == (n, 1)
    assert torch.all(predicted >= 0.)
    assert torch.all(predicted <= 1.)


def test_generator_output():
    c = 1
    d = 8
    gen = Generator(c, d)

    latent_vector  = torch.ones((2, 100))
    sound_generation = gen(latent_vector)
    # (N, C, W)
    assert sound_generation.shape == (2, c, 16384)


def test_phase_shuffle():
    phase_shuffle = PhaseShuffle(7)

    # Check reflecting padding works
    x = torch.arange(10, dtype=torch.float).reshape((1, 1, 10))
    prediction = phase_shuffle(x)
    expected = torch.FloatTensor([2, 3, 4, 5, 6, 7, 8, 9, 8, 7]).reshape((1, 1, 10))

    assert prediction.shape == expected.shape, "Shape not preserved in phase shuffle"
    assert torch.equal(prediction, expected), "Reflection padding did not work"

    # Phase shift not random
    same_prediction = phase_shuffle(x)
    assert not np.array_equal(prediction, same_prediction), "Phase shuffle not random under this seed"


def test_phase_indentity():
    phase_shuffle = PhaseShuffle(0)

    x = torch.arange(10, dtype=torch.float).reshape((1, 1, 10))
    prediction = phase_shuffle(x)
    assert torch.equal(prediction, x), "Phase shuffle identity changed the output"


