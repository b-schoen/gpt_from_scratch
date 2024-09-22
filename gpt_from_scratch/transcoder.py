import dataclasses

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
import torch.optim as optim

import transformer_lens as tl
import math

import einops
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

import transformer_lens

TranscoderIn = Float[Tensor, "... d_in"]
TranscoderHidden = Float[Tensor, "... d_hidden"]
TranscoderOut = Float[Tensor, "... d_out"]

# from https://arxiv.org/html/2406.11944v1#S3 appendix E
# transcoder_expansion_factor = 32
TRANSCODER_EXPANSION_FACTOR = 32


@dataclasses.dataclass
class TranscoderConfig:
    # Input dimension of the transcoder
    d_in: int

    # Output dimension of the transcoder
    d_out: int

    # Hidden dimension of the transcoder
    d_hidden: int

    # Data type for tensor operations
    dtype: torch.dtype

    # Device to run the model on (e.g., 'cpu' or 'cuda')
    device: torch.device

    @classmethod
    def from_model(
        cls,
        model: transformer_lens.HookedTransformer,
        device: torch.device,
        expansion_factor: int = TRANSCODER_EXPANSION_FACTOR,
    ) -> "TranscoderConfig":

        return cls(
            d_in=model.cfg.d_model,
            d_out=model.cfg.d_model,
            # our transcoder has a hidden dimension of d_mlp * expansion factor
            d_hidden=model.cfg.d_mlp * expansion_factor,
            dtype=model.cfg.dtype,
            device=device,
        )


@dataclasses.dataclass
class TranscoderTrainingConfig:

    # Name of the layer to hook into for feature extraction
    #
    # if no layer norm:
    #  - blocks.0.hook_resid_mid
    #  - blocks.0.hook_mlp_out
    # else:
    #  - blocks.0.ln2.hook_normalized
    #  - blocks.0.hook_mlp_out
    hook_point: str
    out_hook_point: str

    num_epochs: int = 100

    # both from https://arxiv.org/html/2406.11944v1#S3 appendix E
    #
    # these are currently from https://github.com/jacobdunefsky/transcoder_circuits/blob/master/train_transcoder.py#L25 though
    # learning_rate: float = 0.0004 * 10
    # l1_coefficient: float = 0.0014 * 10
    learning_rate: float = 2 * 10e-3
    l1_coefficient: float = 5.5 * 10e-3

    @property
    def hook_point_layer(self) -> int:
        "Parse out the hook point layer as int ex: 'blocks.8.ln2.hook_normalized' -> 8"
        return int(self.hook_point.split(".")[1])


@dataclasses.dataclass
class TranscoderResults:
    """
    Dataclass to store the results of the Transcoder forward pass.

    Attributes:
        transcoder_out (Tensor): The output tensor after the transcoder operation.
        hidden_activations (Tensor): Activations from the hidden layer.
    """

    transcoder_out: Float[Tensor, "... d_out"]
    hidden_activations: Float[Tensor, "... d_hidden"]


@dataclasses.dataclass
class TranscoderLoss:
    total_loss: Float[torch.Tensor, ""]
    mse_loss: Float[torch.Tensor, ""]
    l1_loss: Float[torch.Tensor, ""]


class Transcoder(nn.Module):
    """
    Transcoder model for transforming inputs between different representations.

    The Transcoder consists of an encoder, a non-linear activation (ReLU), and a
    decoder.

    This is largely a minimal implementation of [Transcoders Find Interpretable LLM
    Feature Circuits](https://arxiv.org/pdf/2406.11944) and the corresponding
    codebase, which can be found here:

        https://github.com/jacobdunefsky/transcoder_circuits/blob/master/transcoder_training/sparse_autoencoder.py

    (which is in turn mostly from Authur Conmy's https://github.com/ArthurConmy/sae/blob/main/sae/model.py)

    Note:
        For simplicity, we have omitted certain methods related to resampling neurons
        and geometric median initialization.

        Specifically, we've removed:
            - resample_neurons_anthropic
            - collect_anthropic_resampling_losses
            - resample_neurons_l2
            - initialize with geometric median
    """

    def __init__(
        self,
        cfg: TranscoderConfig,
    ) -> None:
        """
        Initialize the Transcoder model.

        Args:
            cfg (TranscoderConfig): Configuration object containing model parameters.
        """
        super().__init__()

        # Store configuration and dimensions from the provided cfg
        self.cfg = cfg
        self.d_in = cfg.d_in  # Input dimension
        self.d_hidden = cfg.d_hidden  # Hidden layer dimension
        self.d_out = cfg.d_out  # Output dimension
        self.dtype = cfg.dtype  # Data type for tensors
        self.device = cfg.device  # Device to run the model on

        # Initialize the encoder weight matrix (W_enc) with Kaiming Uniform initialization
        self.W_enc: Float[Tensor, "d_in d_hidden"] = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.d_in,
                    self.d_hidden,
                    dtype=self.dtype,
                    device=self.device,
                ),
                a=0,
            )
        )

        # Initialize the encoder bias vector (b_enc) with zeros
        self.b_enc: TranscoderHidden = nn.Parameter(
            torch.zeros(
                self.d_hidden,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Initialize the decoder weight matrix (W_dec) with Kaiming Uniform initialization
        self.W_dec: Float[Tensor, "d_hidden d_out"] = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.d_hidden,
                    self.d_out,
                    dtype=self.dtype,
                    device=self.device,
                ),
                a=0,
            )
        )

        # Normalize the decoder weights to have unit norms (following Anthropic's approach)
        with torch.no_grad():
            # Divide each row of W_dec by its L2 norm
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        # Initialize the decoder bias vector for input adjustment (b_dec) with zeros
        self.b_dec: TranscoderIn = nn.Parameter(
            torch.zeros(
                self.d_in,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Initialize the output bias vector (b_dec_out) for the decoder's output with zeros
        self.b_dec_out: TranscoderOut = nn.Parameter(
            torch.zeros(
                self.d_out,
                dtype=self.dtype,
                device=self.device,
            )
        )

    @typechecker  # Enforces type checking of input and output tensors at runtime
    def forward(self, x: TranscoderIn) -> TranscoderResults:
        """
        Perform a forward pass through the Transcoder.

        Args:
            x (Tensor): Input tensor of shape [..., d_in], where '...' represents any number of leading dimensions.

        Returns:
            TranscoderResults: An object containing the output tensor and hidden activations.

        """
        # Ensure the input tensor is of the correct data type
        x = x.to(self.dtype)

        # Adjust the input by subtracting the decoder's bias term (following Anthropic's approach)
        transcoder_in: Float[Tensor, "... d_in"] = x - self.b_dec

        # Compute pre-activation values for the hidden layer (linear transformation)
        # Using einops.einsum for clarity in tensor dimensions
        hidden_pre: Float[Tensor, "... d_hidden"] = (
            einops.einsum(
                transcoder_in,
                self.W_enc,
                "... d_in, d_in d_hidden -> ... d_hidden",
            )
            + self.b_enc  # Add the encoder bias
        )

        # Apply ReLU activation function to introduce non-linearity
        hidden_activations: Float[Tensor, "... d_hidden"] = F.relu(hidden_pre)

        # Compute the output by applying the decoder (another linear transformation)
        transcoder_out: Float[Tensor, "... d_out"] = (
            einops.einsum(
                hidden_activations,
                self.W_dec,
                "... d_hidden, d_hidden d_out -> ... d_out",
            )
            + self.b_dec_out  # Add the decoder's output bias
        )

        # Return the results encapsulated in a TranscoderResults dataclass
        return TranscoderResults(
            transcoder_out=transcoder_out,
            hidden_activations=hidden_activations,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        """
        Normalize the decoder weight matrix so that each neuron's weights have unit L2 norm.

        This method adjusts the decoder weights in-place. It's useful to maintain the norm
        constraints during training or when manually adjusting the model parameters.
        """
        # Normalize each row of W_dec to have a unit L2 norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)


def compute_loss(
    cfg: TranscoderTrainingConfig,
    mlp_out: Float[torch.Tensor, "batch seq d_model"],
    results: TranscoderResults,
) -> TranscoderLoss:
    # Compute MSE loss for each example in the batch
    # We first compute the element-wise squared difference, then average over all dimensions

    # per https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py#L129C54-L129C77
    # and https://github.com/jacobdunefsky/transcoder_circuits/blob/master/sae_training/sparse_autoencoder.py#L147
    mse_loss: Float[torch.Tensor, ""] = F.mse_loss(mlp_out, results.transcoder_out)

    # Compute L1 loss (sparsity regularization) on hidden activations

    # note: doing sum per `https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py#L130`
    l1_loss: Float[torch.Tensor, ""] = cfg.l1_coefficient * results.hidden_activations.abs().sum()

    total_loss = mse_loss + l1_loss

    return TranscoderLoss(total_loss=total_loss, mse_loss=mse_loss, l1_loss=l1_loss)


@dataclasses.dataclass
class TranscoderTrainerOutput:
    loss: TranscoderLoss
    results: TranscoderResults


class TranscoderTrainer:

    def __init__(
        self,
        transcoder_cfg: TranscoderConfig,
        transcoder_training_cfg: TranscoderTrainingConfig,
        device: torch.device,
    ) -> None:
        self.cfg = transcoder_training_cfg
        self.transcoder = Transcoder(transcoder_cfg).to(device)

        # create optimizer
        self.optimizer = torch.optim.AdamW(
            self.transcoder.parameters(),
            lr=transcoder_training_cfg.learning_rate,
        )

        # arbitrary name used to distinguish it in logging
        self._name = f"tc_L{self.cfg.hook_point_layer}"

    @property
    def name(self) -> str:
        return self._name

    def train_on_cache(self, cache: tl.ActivationCache) -> TranscoderTrainerOutput:

        mlp_in = cache[self.cfg.hook_point]
        mlp_out = cache[self.cfg.out_hook_point]

        self.transcoder.train()

        self.optimizer.zero_grad()

        transcoder_results = self.transcoder(mlp_in)

        loss = compute_loss(cfg=self.cfg, mlp_out=mlp_out, results=transcoder_results)

        loss.total_loss.backward()

        self.optimizer.step()

        return TranscoderTrainerOutput(loss=loss, results=transcoder_results)

    def get_wandb_log_dict(
        self,
        trainer_output: TranscoderTrainerOutput,
    ) -> dict[str, float]:
        """
        Creates a dictionary containing relevant statistics for logging to wandb.
        """
        # Extract total loss and individual components
        total_loss = trainer_output.loss.total_loss.item()
        mse_loss = trainer_output.loss.mse_loss.item()
        l1_loss = trainer_output.loss.l1_loss.item()

        # Get learning rate from optimizer
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Calculate norms of encoder and decoder weights
        W_enc_norm = self.transcoder.W_enc.norm().item()
        W_dec_norm = self.transcoder.W_dec.norm().item()

        # Compute statistics about the hidden activations
        hidden_activations = trainer_output.results.hidden_activations

        # Calculate sparsity: fraction of activations near zero
        sparsity = (hidden_activations.abs() < 1e-6).float().mean().item()

        # Calculate mean and std of activations
        mean_activation = hidden_activations.mean().item()
        std_activation = hidden_activations.std().item()

        # Create log dictionary
        log_dict = {
            f"{self._name}/{k}": v
            for k, v in {
                "total_loss": total_loss,
                "mse_loss": mse_loss,
                "l1_loss": l1_loss,
                "learning_rate": learning_rate,
                "W_enc_norm": W_enc_norm,
                "W_dec_norm": W_dec_norm,
                "hidden_activations_sparsity": sparsity,
                "hidden_activations_mean": mean_activation,
                "hidden_activations_std": std_activation,
            }.items()
        }

        return log_dict
