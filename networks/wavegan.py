import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = x.reshape(-1, 256*self.d)
        x = self.dense(x)
        return self.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, c, d, z=100):
        super(Generator, self).__init__()
        self.d = d
        self.stride = 4
        self.kernel_size = 4
        self.z = z

        self.dense=nn.Linear(self.z, 256*d)

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
    def __init__(self, c, d, z=100, alpha=0.2, n=2):
        super(WaveGan, self).__init__()
        self.z = z
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(c, d, z)
        self.discriminator = Discriminator(c, d, alpha, n)

        self.g_optim = optim.Adam(self.generator.parameters())
        self.d_optim = optim.Adam(self.discriminator.parameters())
        self.loss_fn = nn.BCELoss()

        self.to(self.device)

    def train(self, epochs, train_ds):
        writer = SummaryWriter()
        samples = 8
        fixed_noise = torch.randn(samples, self.z, device=self.device)

        for e in tqdm(range(epochs)):

            total_g_loss = 0
            total_d_loss_real = 0
            total_d_loss_fake = 0
            iters = 0

            for batch in train_ds:
                sounds = batch.to(self.device)
                batch_size = sounds.size(0)

                d_loss_real, d_loss_fake = self.train_discriminator(sounds)
                g_loss = self.train_generator(batch_size)

                total_g_loss += g_loss
                total_d_loss_fake += d_loss_fake
                total_d_loss_real += d_loss_real
                iters += batch_size


            # Log fake samples to tensorboard
            with torch.no_grad():
                fake_audio = self.generator(fixed_noise).detach().cpu()

            for i, audio in enumerate(fake_audio):
                writer.add_audio(f"sample_{i}", audio, global_step=e, sample_rate=16384)

            writer.add_scalar("discriminator_loss_real", total_d_loss_real / iters, e)
            writer.add_scalar("discriminator_loss_fake", total_d_loss_fake / iters, e)
            writer.add_scalar("generator_loss", total_g_loss / iters, e)

    def train_generator(self, batch_size):
        # Generator loss calculation
        self.generator.zero_grad()

        noise = torch.randn(batch_size, self.z, device=self.device)
        expected_label = torch.ones((batch_size), dtype=torch.float, device=self.device)
        generated = self.generator(noise)
        classification = self.discriminator(generated)

        loss = self.loss_fn(torch.squeeze(classification), expected_label)
        loss.backward()
        self.g_optim.step()

        return loss.item()

    def train_discriminator(self, real_samples):
        batch_size = real_samples.size(0)
        self.discriminator.zero_grad()

        real_labels = torch.ones((batch_size), dtype=torch.float, device=self.device)
        pred_real = self.discriminator(real_samples)
        loss_real = self.loss_fn(torch.squeeze(pred_real), real_labels)

        latent_vec = torch.randn(batch_size, self.z, device=self.device)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)

        fake_labels = torch.zeros((batch_size), dtype=torch.float, device=self.device)

        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.loss_fn(torch.squeeze(pred_fake), fake_labels)

        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.d_optim.step()

        return loss_real.item(), loss_fake.item()
