import torch
import torch.nn as nn
import torch.nn.functional as F
import jaxtyping
from typing import Literal, Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import wandb

# Import custom types and functions
import jaxtyping
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

import einops
from einops.layers.torch import Rearrange

import numpy as np


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Mean = Float[torch.Tensor, "b d_latent"]
#
# note: we use `logsigma` instead of `sigma` solely because it's more numerically stable
#       - this seems to be a general pattern when we see log
#
LogStdDev = Float[torch.Tensor, "b d_latent"]


# note:
#  - P_theta(x|z) := probabilistic decoder
#  - Q_phi(z|x)   := probabilistic encoder
#
#  - z ~ Q_phi(z|x)
#
#  - `sample_latent_vector` is essentially our "encoder" step
#
class VAE(nn.Module):

    # note: encoder is the same
    def __init__(self, latent_dim_size: int, hidden_dim_size: int) -> None:
        super().__init__()

        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

        # Input shape: (batch_size, 1, 28, 28)

        conv_stride = 2
        conv_kernel_size = 4
        conv_padding = 1

        encoder_conv1_out_channels = 16

        self.encoder_conv1 = nn.Conv2d(
            in_channels=1,  # MNIST images have 1 channel
            out_channels=encoder_conv1_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )

        # Shape after first conv: (batch_size, 16, 14, 14)

        self.encoder_conv2 = nn.Conv2d(
            in_channels=self.encoder_conv1.out_channels,
            out_channels=self.encoder_conv1.out_channels * conv_stride,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
        )

        self.encoder = nn.Sequential(
            # First convolutional block
            self.encoder_conv1,
            nn.ReLU(),
            # Second convolutional block
            self.encoder_conv2,
            nn.ReLU(),
            # Shape after second conv: (batch_size, 32, 7, 7)
            #
            # -> (1, 28, 28) -> (16, 14, 14) -> (32, 7, 7)
            #
            # halving by stride length
            Rearrange("b c h w -> b (c h w)"),
            # Shape after flatten: (batch_size, 32 * 7 * 7) = (batch_size, 1568)
            # two fully connected layers with ReLU in between them
            # go from the flattened dimension to the hidden dimension (can be arbitrarily large?)
            nn.Linear(
                in_features=self.encoder_conv2.out_channels * 7 * 7,
                out_features=hidden_dim_size,
            ),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(
            in_features=hidden_dim_size,
            out_features=latent_dim_size,
        )

        self.logsigma_layer = nn.Linear(
            in_features=hidden_dim_size,
            out_features=latent_dim_size,
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            # now we have a (batch_size, latent_dim_size)
            # first go back to the hidden dimension
            nn.Linear(in_features=latent_dim_size, out_features=hidden_dim_size),
            nn.ReLU(),
            # map from hidden dim to number needed to invert conv
            nn.Linear(
                in_features=hidden_dim_size,
                out_features=self.encoder_conv2.out_channels * 7 * 7,
            ),
            #
            # need to reshape it for the transposed convolution
            # - reshape from (batch_size, hidden_dim_size) to (batch_size, 32, 7, 7)
            #
            Rearrange(
                "b (c h w) -> b c h w",
                c=self.encoder_conv2.out_channels,
                h=7,
                w=7,
            ),
            # now undo the conv
            nn.ConvTranspose2d(
                in_channels=self.encoder_conv2.out_channels,
                out_channels=self.encoder_conv1.out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=self.encoder_conv1.out_channels,
                out_channels=1,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            ),
        )

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def sample_latent_vector(
        self,
        x: Float[torch.Tensor, "b c h w"],
    ) -> tuple[Float[torch.Tensor, "b d_latent"], Mean, LogStdDev]:
        """
        Passes `x` through the encoder. Returns the mean and log std dev of the latent vector,
        as well as the latent vector itself. This function can be used in `forward`, but also
        used on its own to generate samples for evaluation.
        """

        encoded_hidden: Float[torch.Tensor, "b d_hidden"] = self.encoder(x)

        mu: Float[torch.Tensor, "b d_latent"] = self.mu_layer(encoded_hidden)
        logsigma: Float[torch.Tensor, "b d_latent"] = self.logsigma_layer(
            encoded_hidden
        )

        # recover sigma
        sigma = logsigma.exp()

        # sample `epsilon` from normal distribution
        epsilon: Float[torch.Tensor, "b d_latent"] = torch.normal(
            mean=torch.zeros_like(mu),
            std=torch.ones_like(sigma),
        )

        # z := mu + sigma * epsilon
        # <latent> = <average> + <stddev> * <sampled>
        #
        # where these are elementwise operations (as this is a reparameterization)
        z: Float[torch.Tensor, "b d_latent"] = mu + sigma * epsilon

        return (z, mu, logsigma)

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
    ) -> tuple[Float[torch.Tensor, "b c h w"], Mean, LogStdDev]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well
        as mu and logsigma.
        """

        z, mu, logsigma = self.sample_latent_vector(x)

        x_prime: Float[torch.Tensor, "b c h w"] = self.decoder(z)

        return (x_prime, mu, logsigma)


# N(mu, sigma^2) || N(0, 1)
#
# => We know this should be normal because epsilon
# => Actually it should learn to be normal
#
# -> note:
#  - ELBO = reconstruction_loss_elbo - kl_div_elbo
#
#  - reconstruction loss elbo => reconstruction loss between z <-> x
#
#  - kl_div_elbo => div between dist of latent vectors produced vs dist of latent vectors
#                   i.e.    (mu(x), sigma(x)) vs (0, 1)
#
def compute_kl_divergence_against_normal_distribution(
    mu: Float[torch.Tensor, "b d_latent"],
    logsigma: Float[torch.Tensor, "b d_latent"],
) -> Float[torch.Tensor, ""]:

    # simplifies down to

    sigma_squared = torch.exp(2 * logsigma)
    mu_squared = mu**2

    kl_div = (((sigma_squared + mu_squared) - 1) / 2) - logsigma

    # average over batch and latent dim (same as MSE does in practice)
    #
    return kl_div.mean()


@dataclass
class VAELoss:
    total_loss: Float[torch.Tensor, "b"]
    reconstruction_loss: Float[torch.Tensor, "b"]
    kl_divergence_loss: Float[torch.Tensor, "b"]
    mu: Mean
    sigma: Float[torch.Tensor, "b"]


def compute_vae_loss(
    input: Float[torch.Tensor, "b c h w"],
    reconstructed_input: Float[torch.Tensor, "b c h w"],
    mu: Mean,
    logsigma: LogStdDev,
    beta_kl: float,
) -> VAELoss:

    reconstruction_loss = F.mse_loss(input, reconstructed_input)

    kl_divergence_loss = compute_kl_divergence_against_normal_distribution(
        mu=mu,
        logsigma=logsigma,
    )

    # scale KL divergence
    kl_divergence_loss = kl_divergence_loss * beta_kl

    total_loss = reconstruction_loss + kl_divergence_loss

    return VAELoss(
        total_loss,
        reconstruction_loss,
        kl_divergence_loss,
        mu=mu,
        sigma=logsigma.exp(),
    )


@dataclass
class VAEArgs:
    latent_dim_size: int = 16
    hidden_dim_size: int = 512
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-4
    betas: tuple[float] = (0.9, 0.999)

    # how much to penalize KL divergence
    beta_kl: float = 0.1
    seconds_between_eval: int = 5
    wandb_project: Optional[str] = "vae"
    wandb_name: Optional[str] = None

    # Add train and validation datasets to args
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset


class VAETrainer:
    def __init__(self, args: VAEArgs) -> None:

        self.args = args

        self.trainset = args.train_dataset  # Use train dataset from args

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        # Create validation dataloader
        self.valloader = DataLoader(
            args.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=args.betas,
        )

        self.step = 0

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def training_step(self, img: Float[torch.Tensor, "b c h w"]) -> VAELoss:
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        """

        self.optimizer.zero_grad()

        reconstructed_img, mu, logsigma = self.model(img)

        # calculate loss on the reconstructed image
        vae_loss = compute_vae_loss(
            img,
            reconstructed_img,
            mu,
            logsigma,
            self.args.beta_kl,
        )

        # propagate loss backwards
        vae_loss.total_loss.backward()

        # step the optimizer
        self.optimizer.step()

        # return the loss so that we can log it
        return vae_loss

    @torch.inference_mode()
    def evaluate(self) -> None:
        """
        Evaluates model on validation data, logs to weights & biases.
        """

        self.model.eval()

        print(f"[{self.step=}] Evaluating on validation data...")

        # Use the first batch from validation data for logging
        val_batch = next(iter(self.valloader))
        val_img, _ = val_batch
        val_img = val_img.to(device)

        reconstructed_val, mu, logsigma = self.model(val_img)

        vae_loss = compute_vae_loss(
            val_img,
            reconstructed_val,
            mu,
            logsigma,
            self.args.beta_kl,
        )

        def reshape_for_wandb(x):
            # Rearrange from (b, c, h, w) to (b, h, w, c)
            x = einops.rearrange(x, "b c h w -> b h w c")

            # Ensure the values are in the range [0, 1]
            x = (x - x.min()) / (x.max() - x.min())

            # Convert to numpy array and multiply by 255 to get values in [0, 255]
            x = (x.cpu().numpy() * 255).astype(np.uint8)

            # Create a list of wandb.Image objects
            return [wandb.Image(img) for img in x]

        wandb.log(
            {
                "val_images": reshape_for_wandb(val_img),
                "reconstructed_val_images": reshape_for_wandb(reconstructed_val),
            },
            step=self.step,
        )

        # put model back in train mode
        self.model.train()

        # log vae loss
        self._log_vae_loss(prefix="val", vae_loss=vae_loss)

        print(f"[{self.step=}] val_loss: {vae_loss.total_loss.item():.6f}")

    def _log_vae_loss(
        self,
        prefix: Literal["train"] | Literal["val"],
        vae_loss: VAELoss,
    ) -> None:

        wandb.log(
            {
                f"{prefix}_total_loss": vae_loss.total_loss.item(),
                f"{prefix}_reconstruction_loss": vae_loss.reconstruction_loss.item(),
                f"{prefix}_kl_divergence_loss": vae_loss.kl_divergence_loss.item(),
                f"{prefix}_mu": vae_loss.mu.mean().item(),
                f"{prefix}_sigma": vae_loss.sigma.mean().item(),
            },
            step=self.step,
        )

    def train(self) -> None:
        """
        Performs a full training run, logging to wandb.
        """

        last_log_time = time.time()

        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for i, (img, label) in enumerate(
                progress_bar
            ):  # remember that label is not used

                img = img.to(device)

                vae_loss = self.training_step(img)

                # log vae loss
                self._log_vae_loss(prefix="train", vae_loss=vae_loss)

                # Evaluate model on the validation data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

                # Update `step`, which is used for wandb reporting
                self.step += img.shape[0]

                # Update progress bar
                progress_bar.set_description(
                    f"{epoch=}, {vae_loss.total_loss.item()=:.4f}, "
                    f"examples_seen={self.step}"
                )

        wandb.finish()
