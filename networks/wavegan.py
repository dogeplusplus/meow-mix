import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseShuffle(nn.Module):
    def __init__(self, n):
        super(PhaseShuffle, self).__init__()
        self.n = n

    def forward(self, x):
        shift = torch.randint(-self.n, self.n+1, (1,))
        # Reflection pad to the right
        if shift <= 0:
            right_pad = abs(shift)
            x = F.pad(x, (0, right_pad), mode="reflect")[..., right_pad:]
        else:
            left_pad = shift
            x = F.pad(x, (left_pad, 0), mode="reflect")[..., :-left_pad]

        return x

class Discriminator(nn.Module):
    def __init__(self, c, d, alpha=0.2, n=2):
        super(Discriminator, self).__init__()
        self.d = d

        self.kernel_size = 2
        self.stride = 4

        self.base = nn.Sequential(
            nn.Conv1d(c, d, self.kernel_size, self.stride),
            nn.LeakyReLU(alpha),
            PhaseShuffle(n),

            nn.Conv1d(d, 2*d, self.kernel_size, self.stride),
            nn.LeakyReLU(alpha),
            PhaseShuffle(n),

            nn.Conv1d(2*d, 4*d, self.kernel_size, self.stride),
            nn.LeakyReLU(alpha),
            PhaseShuffle(n),

            nn.Conv1d(4*d, 8*d, self.kernel_size, self.stride),
            nn.LeakyReLU(alpha),
            PhaseShuffle(n),

            nn.Conv1d(8*d, 16*d, self.kernel_size, self.stride),
            nn.LeakyReLU(alpha),
            PhaseShuffle(n),
        )

        self.dense = nn.Linear(256*d, 1)

    def forward(self, x):
        x = self.base(x)
        x = x.reshape(-1, 256*self.d)
        return self.dense(x)


class Generator(nn.Module):
    def __init__(self, c, d):
        super(Generator, self).__init__()
        self.d = d
        self.stride = 4
        self.kernel_size = 4

        self.dense=nn.Linear(100, 256*d)

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(16*d, 8*d, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.ConvTranspose1d(8*d, 4*d, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.ConvTranspose1d(4*d, 2*d, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.ConvTranspose1d(2*d, d, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.ConvTranspose1d(d, c, self.kernel_size, self.stride),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 16*self.d, 16)
        return self.layers(x)


class WaveGan(nn.Module):
    def __init__(self, c, d, alpha=0.2, n=2):
        self.generator = Generator(c, d)
        self.discriminator = Discriminator(c, d, alpha, n)


if __name__ == "__main__":
    main()
