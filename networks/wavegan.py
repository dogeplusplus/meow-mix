import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from os.path import join
from tqdm import tqdm
from datetime import datetime
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


def gradient_penalty(discriminator, device, real_samples, fake_samples):
    """
    Wasserstein Gradient Penalty Loss

    Parameters
    ----------
    discriminator : nn.Module
        GAN discriminator
    device : str
        GPU or CPU device
    real_samples : torch.FloatTensor
        Real samples from the training dataset
    fake_samples : torch.FloatTensor
        Samples produced by the generator

    Returns
    ----------
    gradient_penalty: torch.FloatTensor
        Gradient penalty of the real and fake audio samples
    """
    alpha = torch.randn(real_samples.size(0), 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.shape[0], dtype=torch.float, requires_grad=False).to(device)

    gradients = torch.autograd.grad(
        outputs=torch.squeeze(d_interpolates),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


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

@dataclass
class ModelConfig:
    n: int = 2
    d: int = 1
    c: int = 1
    z: int = 100
    stride: int = 4
    kernel_size: int = 4
    alpha: float = 0.2


@dataclass
class TrainingConfig:
    model_config: ModelConfig
    epochs: int = 100
    learning_rate: float = 1e-3
    lambda_gp: float = 10
    current_epoch: int = 0
    save_every: int = 20
    log_dir: str = ""


class WaveGan(nn.Module):
    def __init__(self, train_config: TrainingConfig):
        super(WaveGan, self).__init__()
        self.train_config = train_config
        self.model_config = train_config.model_config


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(self.model_config.c, self.model_config.d, self.model_config.z)
        self.discriminator = Discriminator(self.model_config.c, self.model_config.d, self.model_config.alpha, self.model_config.n)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.g_optim = optim.Adam(self.generator.parameters())
        self.d_optim = optim.Adam(self.discriminator.parameters())
        self.loss_fn = nn.BCELoss()
 
        if train_config.log_dir == "":
            train_config.log_dir = join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            # Load previous session if exists
            generator_path = join(train_config.log_dir, "generator_latest.pt")
            discriminator_path = join(train_config.log_dir, "discriminator_latest.pt")
            opt_g_path = join(train_config.log_dir, "gen_optim.pt")
            opt_d_path = join(train_config.log_dir, "disc_optim.pt")

            self.generator.load_state_dict(torch.load(generator_path))
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            self.g_optim.load_state_dict(torch.load(opt_g_path))
            self.d_optim.load_state_dict(torch.load(opt_d_path))


    def train(self, train_ds):
        config = self.train_config
        os.makedirs(config.log_dir, exist_ok=True)

        writer = SummaryWriter(config.log_dir)

        samples = 8
        fixed_noise = torch.randn(samples, self.model_config.z, device=self.device)

        for e in range(config.current_epoch, config.current_epoch + config.epochs):
            total_g_loss = 0
            total_d_loss_real = 0
            total_d_loss_fake = 0
            total_accuracy = 0
            iters = 0

            description = f"Epoch {e}"
            train_bar = tqdm(train_ds, desc=description)

            for batch in train_bar:
                real = batch.to(self.device)
                batch_size = real.size(0)

                d_loss_real, d_loss_fake, accuracy = self.train_discriminator(real)
                g_loss = self.train_generator(batch_size)

                total_accuracy += accuracy
                total_g_loss += g_loss
                total_d_loss_fake += d_loss_fake
                total_d_loss_real += d_loss_real
                iters += batch_size

                live_metrics = {
                    "d_loss_real": f"{total_d_loss_real / iters:0.5f}",
                    "d_loss_fake": f"{total_d_loss_fake / iters:0.5f}",
                    "g_loss": f"{total_g_loss / iters:0.5f}",
                    "accuracy": f"{total_accuracy / iters:0.5f}",
                }

                train_bar.set_postfix(live_metrics)


            # Log fake samples to tensorboard
            with torch.no_grad():
                fake_audio = self.generator(fixed_noise).detach().cpu()

            for i, audio in enumerate(fake_audio):
                writer.add_audio(f"sample_{i}", audio, global_step=e, sample_rate=16384)

            writer.add_scalar("discriminator_loss_real", total_d_loss_real / iters, e)
            writer.add_scalar("discriminator_loss_fake", total_d_loss_fake / iters, e)
            writer.add_scalar("generator_loss", total_g_loss / iters, e)

            if e % config.save_every == 0:
                torch.save(self.generator.state_dict(), os.path.join(config.log_dir, f"generator_{e}.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join(config.log_dir, f"discriminator_{e}.pt"))

                torch.save(self.generator.state_dict(), os.path.join(config.log_dir, f"generator_latest.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join(config.log_dir, f"discriminator_latest.pt"))
                torch.save(self.g_optim.state_dict(), os.path.join(config.log_dir, f"gen_optim.pt"))
                torch.save(self.d_optim.state_dict(), os.path.join(config.log_dir, f"disc_optim.pt"))

        self.train_config.current_epoch += self.train_config.epochs

        with open(join(config.log_dir, "train_config.json"), "w") as f:
            json.dump(asdict(self.train_config), f, indent=4)
        with open(join(config.log_dir, "model_config.json"), "w") as f:
            json.dump(asdict(self.model_config), f, indent=4)


    def train_generator(self, batch_size):
        # Generator loss calculation
        self.generator.zero_grad()

        noise = torch.randn(batch_size, self.model_config.z, device=self.device)
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

        latent_vec = torch.randn(batch_size, self.model_config.z, device=self.device)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)

        fake_labels = torch.zeros((batch_size), dtype=torch.float, device=self.device)

        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.loss_fn(torch.squeeze(pred_fake), fake_labels)

        gp_loss = gradient_penalty(self.discriminator, self.device, real_samples, fake_samples)

        loss = loss_real + loss_fake + self.train_config.lambda_gp * gp_loss
        loss.backward()
        self.d_optim.step()

        combined_prediction = torch.cat([pred_real, pred_fake], dim=0) > 0.5
        combined_labels = torch.cat([real_labels, fake_labels], dim=0)
        accuracy = torch.mean(torch.eq(combined_prediction, combined_labels).type(torch.FloatTensor).to(self.device))

        return loss_real.item(), loss_fake.item(), accuracy.item()


