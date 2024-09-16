import dataclasses

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
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
    ) -> "TranscoderConfig":

        return cls(
            d_in=model.cfg.d_model,
            d_out=model.cfg.d_model,
            # our transcoder has a hidden dimension of d_mlp * expansion factor
            d_hidden=model.cfg.d_mlp * TRANSCODER_EXPANSION_FACTOR,
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


@dataclasses.dataclass
class TranscoderLoss:
    mse_loss: Float[torch.Tensor, ""]
    l1_loss: Float[torch.Tensor, ""]


def compute_loss(
    cfg: TranscoderTrainingConfig,
    mlp_out: Float[torch.Tensor, "batch seq d_model"],
    results: TranscoderResults,
) -> TranscoderLoss:
    # Compute MSE loss for each example in the batch
    # We first compute the element-wise squared difference, then average over all dimensions
    mse_loss: Float[torch.Tensor, ""] = torch.mean(
        (results.transcoder_out - mlp_out) ** 2
    )

    # Double-check: This calculation is correct
    # Explanation:
    # 1. (results.transcoder_out - mlp_out) computes the difference between predicted and actual values
    # 2. ** 2 squares these differences
    # 3. torch.mean() averages over all dimensions (batch, seq, and d_model)
    # This is the standard formula for Mean Squared Error (MSE) loss

    # Compute L1 loss (sparsity regularization) on hidden activations
    l1_loss: Float[torch.Tensor, ""] = (
        cfg.l1_coefficient * torch.abs(results.hidden_activations).mean()
    )

    return TranscoderLoss(mse_loss=mse_loss, l1_loss=l1_loss)
