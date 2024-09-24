import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import transformer_lens as tl

from jaxtyping import Float

import dataclasses


@dataclasses.dataclass
class SAEConfig:

    input_size: int
    """input_size: Dimensionality of input data"""

    n_dict_components: int
    """n_dict_components: Number of dictionary components"""

    init_decoder_orthogonal: bool = True
    """init_decoder_orthogonal: Initialize the decoder weights to be orthonormal"""


@dataclasses.dataclass
class SAEOutput:
    x_sae_activations: Float[torch.Tensor, "batch n_dict_components"]
    x_reconstructed: Float[torch.Tensor, "batch input_size"]


@dataclasses.dataclass
class SAELossConfig:
    l1_coefficient: float = 1.0


@dataclasses.dataclass
class SAELoss:
    total_loss: Float[torch.Tensor, ""]
    reconstruction_loss: Float[torch.Tensor, ""]
    sparsity_loss: Float[torch.Tensor, ""]


@dataclasses.dataclass
class SAETrainerConfig:
    hook_point: str
    lr: float = 1e-3
    num_epochs: int = 100
    eval_every_n_steps: int = 10
    loss_config: SAELossConfig = dataclasses.field(default_factory=SAELossConfig)


class SAE(nn.Module):
    """
    Sparse AutoEncoder

    There are many similar implementations. Here we just use the e2e SAE ones in case it makes it
    easier to integrate with later code they use.

    From:
    - https://github.com/ApolloResearch/e2e_sae/blob/main/e2e_sae/models/sparsifiers.py#L10

    See also:
    - https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py

    """

    def __init__(self, cfg: SAEConfig) -> None:

        super().__init__()

        self.cfg = cfg
        self.n_dict_components = cfg.n_dict_components
        self.input_size = cfg.input_size

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.input_size, cfg.n_dict_components, dtype=torch.float)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.n_dict_components, cfg.input_size, dtype=torch.float)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_dict_components, dtype=torch.float))
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_size, dtype=torch.float))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x: Float[torch.Tensor, "batch input_size"]) -> SAEOutput:
        """Pass input through the encoder and normalized decoder."""

        # center x {per anthropic} same as transcoder
        x_cent = x - self.b_dec

        x_sae_activations: Float[torch.Tensor, "batch n_dict_components"] = F.relu(
            x_cent @ self.W_enc + self.b_enc
        )

        x_reconstructed: Float[torch.Tensor, "batch input_size"] = (
            x_sae_activations @ self.W_dec + self.b_dec
        )

        return SAEOutput(x_reconstructed=x_reconstructed, x_sae_activations=x_sae_activations)

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# TODO(bschoen): https://github.com/ApolloResearch/e2e_sae/blob/main/e2e_sae/losses.py is a REALLY
#                good reference here
def compute_loss(
    x: Float[torch.Tensor, "batch input_size"],
    x_sae_output: SAEOutput,
    sae_loss_cfg: SAELossConfig,
) -> SAELoss:

    x_reconstructed = x_sae_output.x_reconstructed

    reconstruction_loss = F.mse_loss(x, x_reconstructed)

    sparsity_loss = sae_loss_cfg.l1_coefficient * x_sae_output.x_sae_activations.abs().sum()

    return SAELoss(
        total_loss=reconstruction_loss + sparsity_loss,
        reconstruction_loss=reconstruction_loss,
        sparsity_loss=sparsity_loss,
    )


@dataclasses.dataclass
class SAETrainerOutput:
    loss: SAELoss
    results: SAEOutput


class SAETrainer:

    def __init__(
        self,
        sae_cfg: SAEConfig,
        sae_trainer_cfg: SAETrainerConfig,
        device: torch.device,
    ) -> None:

        self.cfg = sae_trainer_cfg

        self.sae = SAE(sae_cfg).to(device)

        # create optimizer
        self.optimizer = torch.optim.AdamW(
            self.sae.parameters(),
            lr=sae_trainer_cfg.lr,
        )

        # create learning rate scheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.num_epochs)

        # arbitrary name used to distinguish it in logging

        short_layer_name = sae_trainer_cfg.hook_point.split(".")[1]
        short_hook_point = sae_trainer_cfg.hook_point.split(".")[-1]

        self._name = f"sae_L{short_layer_name}_{short_hook_point}"

    @property
    def name(self) -> str:
        return self._name

    def train_on_cache(self, cache: tl.ActivationCache) -> SAETrainerOutput:
        x = cache[self.cfg.hook_point]

        self.sae.train()

        self.optimizer.zero_grad()

        sae_output = self.sae(x)

        loss = compute_loss(x=x, x_sae_output=sae_output, sae_loss_cfg=self.cfg.loss_config)

        loss.total_loss.backward()

        self.sae.make_decoder_weights_and_grad_unit_norm()

        self.optimizer.step()

        self.scheduler.step()

        return SAETrainerOutput(loss=loss, results=sae_output)

    def get_wandb_log_dict(self, trainer_output: SAETrainerOutput) -> dict[str, float]:
        """
        Creates a dictionary containing relevant statistics for logging to wandb.
        """
        # Extract total loss and individual components
        total_loss = trainer_output.loss.total_loss.item()
        reconstruction_loss = trainer_output.loss.reconstruction_loss.item()
        sparsity_loss = trainer_output.loss.sparsity_loss.item()

        # Get learning rate from optimizer
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Calculate norms of encoder and decoder weights
        encoder_weight_norm = self.sae.W_enc.norm().item()
        decoder_weight_norm = self.sae.W_dec.norm().item()

        # Optionally calculate sparsity and mean of activations
        x_sae_activations = trainer_output.results.x_sae_activations

        # Calculate sparsity: fraction of activations near zero
        sparsity_threshold = 1e-6  # You can adjust this threshold
        sparsity = (x_sae_activations.abs() < sparsity_threshold).float().mean().item()

        # Calculate mean and std of activations
        mean_activation = x_sae_activations.mean().item()
        std_activation = x_sae_activations.std().item()

        # Create log dictionary
        log_dict = {
            f"{self._name}/{k}": v
            for k, v in {
                "total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "sparsity_loss": sparsity_loss,
                "learning_rate": learning_rate,
                "encoder_weight_norm": encoder_weight_norm,
                "decoder_weight_norm": decoder_weight_norm,
                "activations_sparsity": sparsity,
                "activations_mean": mean_activation,
                "activations_std": std_activation,
            }.items()
        }

        return log_dict
