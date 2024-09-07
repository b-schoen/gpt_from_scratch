# %%

# Enable autoreload to automatically reload modules when they change

from IPython import get_ipython

# do this so that formatter not messed up
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# Enable jaxtyping based typechecking
import jaxtyping
import typeguard

# Inline comment: This magic command enables runtime type checking using jaxtyping and typeguard
# ipython.run_line_magic("load_ext", "jaxtyping")

# Inline comment: This sets the typecheck mode to 'jaxtyping', which allows for more precise tensor shape checking
# ipython.run_line_magic("jaxtyping.typechecker", "typeguard.typechecked")


import jaxtyping
from typeguard import typechecked as typechecker

# Inline comment: Using IPython API to call magic commands programmatically
# This achieves the same effect as the magic commands, but allows for more flexibility and control

# Inline comment: This magic command allows for automatic reloading of imported modules,
# which is particularly useful during development and debugging


# %%

import os
import sys
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Optional, Tuple, List, Literal, Union
import plotly.express as px
import torchinfo
import time
import wandb
from PIL import Image
import pandas as pd
from pathlib import Path
from datasets import load_dataset, DatasetDict

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_gans_and_vaes"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part2_cnns.utils import print_param_count
import part5_gans_and_vaes.tests as tests
import part5_gans_and_vaes.solutions as solutions
from plotly_utils import imshow

from part2_cnns.solutions import (
    Linear,
    ReLU,
    Sequential,
    BatchNorm2d,
)
from part2_cnns.solutions_bonus import (
    pad1d,
    pad2d,
    conv1d_minimal,
    conv2d_minimal,
    Conv2d,
    Pair,
    IntOrPair,
    force_pair,
)

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available() else "cpu"
)

MAIN = __name__ == "__main__"
# %%

celeb_data_dir = section_dir / "data/celeba/img_align_celeba"

if celeb_data_dir.exists():
    print("Dataset already loaded.")
else:
    dataset: DatasetDict = load_dataset("nielsr/CelebA-faces")
    print("Dataset loaded.")

    os.makedirs(celeb_data_dir)
    for idx, item in tqdm(
        enumerate(dataset["train"]),
        total=len(dataset["train"]),
        desc="Saving individual images...",
    ):
        # The image is already a JpegImageFile, so we can directly save it
        item["image"].save(
            exercises_dir
            / f"part5_gans_and_vaes/data/celeba/img_align_celeba/{idx:06}.jpg"
        )
    print("All images have been saved.")


# %%

import enum


class DatasetType(enum.Enum):
    MNIST = enum.auto()
    CELEB = enum.auto()


# Save the dataset to a file
import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt


def generate_shape(shape_type, size=28, thickness=2):
    canvas = np.zeros((size, size))
    center = size // 2

    if shape_type == "circle":
        y, x = np.ogrid[-center : size - center, -center : size - center]
        mask = (x * x + y * y <= (center - thickness) ** 2) & (
            x * x + y * y >= (center - 2 * thickness) ** 2
        )
    elif shape_type == "square":
        mask = np.zeros((size, size), dtype=bool)
        mask[
            center - center // 2 : center + center // 2,
            center - center // 2 : center + center // 2,
        ] = True
        inner_mask = np.zeros((size, size), dtype=bool)
        inner_mask[
            center - center // 2 + thickness : center + center // 2 - thickness,
            center - center // 2 + thickness : center + center // 2 - thickness,
        ] = True
        mask = mask & ~inner_mask
    elif shape_type == "triangle":
        y, x = np.ogrid[-center : size - center, -center : size - center]
        mask = (y >= -x) & (y >= x) & (y <= center - thickness)
        inner_mask = (
            (y >= -x + thickness) & (y >= x + thickness) & (y <= center - 2 * thickness)
        )
        mask = mask & ~inner_mask

    canvas[mask] = 1
    return canvas


def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def generate_shapes_dataset(num_samples=100000, size=28):
    shapes = ["circle", "square", "triangle"]
    dataset = []
    labels = []  # List to store shape labels

    for _ in tqdm(range(num_samples)):
        shape_type = np.random.choice(shapes)
        thickness = np.random.randint(1, 4)
        shape = generate_shape(shape_type, size, thickness)

        # Random rotation
        angle = np.random.uniform(0, 360)
        shape = np.rot90(shape, k=int(angle // 90))

        # Random scaling
        scale = np.random.uniform(0.5, 1.5)
        shape = np.clip(shape * scale, 0, 1)

        # Add noise
        shape = add_noise(shape, noise_level=0.1)

        dataset.append(shape)
        labels.append(shapes.index(shape_type))  # Store the label as an integer

    return np.array(dataset), np.array(labels)


# To save:
# np.save('shapes_dataset.npy', combined_data)

# To load:
# loaded_data = np.load('shapes_dataset.npy', allow_pickle=True)
# dataset, labels = zip(*loaded_data)


# Function to load the dataset
def load_numpy_dataset(file_path):
    if os.path.exists(file_path):
        loaded_dataset = np.load(file_path)
        print(f"Dataset loaded from {file_path}")
        print(f"Loaded dataset shape: {loaded_dataset.shape}")
        return loaded_dataset
    else:
        print(
            f"File {file_path} not found. Please generate and save the dataset first."
        )
        return None


# %%


# Define the file path
shapes_dataset_filepath = "shape_dataset_data.npy"
shapes_dataset_labels_filepath = "shape_dataset_labels.npy"

if not os.path.exists(shapes_dataset_filepath):

    # Generate the dataset
    shapes_dataset, shapes_dataset_labels = generate_shapes_dataset()

    # Save the dataset
    np.save(shapes_dataset_filepath, shapes_dataset)
    print(f"Dataset saved to {shapes_dataset_filepath}")

    np.save(shapes_dataset_labels_filepath, shapes_dataset_labels)
    print(f"Dataset labels saved to {shapes_dataset_labels_filepath}")


# %%


def get_dataset(
    dataset: Literal["MNIST", "CELEB", "SHAPES"], train: bool = True
) -> Dataset:
    assert dataset in ["MNIST", "CELEB", "SHAPES"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.ImageFolder(
            root=exercises_dir / "part5_gans_and_vaes/data/celeba", transform=transform
        )

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = datasets.MNIST(
            root=exercises_dir / "part5_gans_and_vaes/data",
            transform=transform,
            download=True,
        )

    elif dataset == "SHAPES":
        print("Loading shapes dataset...")
        shapes_images = np.load(shapes_dataset_filepath)
        shapes_labels = np.load(shapes_dataset_labels_filepath)

        # Convert to tensor and add channel dimension to match MNIST format
        shapes_tensor = t.from_numpy(shapes_images).float().unsqueeze(1)
        labels_tensor = t.from_numpy(shapes_labels).long()

        # Normalize to match MNIST statistics
        shapes_tensor = (shapes_tensor - shapes_tensor.mean()) / shapes_tensor.std()

        # Create a dataset with actual labels
        trainset = t.utils.data.TensorDataset(shapes_tensor, labels_tensor)

        print(
            f"Loaded shapes dataset: {len(trainset)} samples, shape: {shapes_tensor.shape[1:]}, {len(set(labels_tensor))} unique labels"
        )

    return trainset


# %%

trainset: Dataset = get_dataset("SHAPES", train=True)

# %%


def display_data(x: t.Tensor, nrows: int, title: str) -> None:
    """Displays a batch of data, using plotly."""

    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()

    # Normalize, in the 0-1 range
    y = (y - y.min()) / (y.max() - y.min())

    # Display data
    imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 5),
        title=title + f"<br>single input shape = {x[0].shape}",
    )


# Load in MNIST, get first batch from dataloader, and display
trainset_mnist = get_dataset("MNIST")
x = next(iter(DataLoader(trainset_mnist, batch_size=64)))[0]
display_data(x, nrows=8, title="MNIST data")

# Load in CelebA, get first batch from dataloader, and display
trainset_celeb = get_dataset("CELEB")
x = next(iter(DataLoader(trainset_celeb, batch_size=64)))[0]
display_data(x, nrows=8, title="CalebA data")

trainset_shapes = get_dataset("SHAPES")
x = next(iter(DataLoader(trainset_shapes, batch_size=64)))[0]
display_data(x, nrows=8, title="Shapes data")
# %%

# note: this is just VAL

from jaxtyping import Float, Int

testset: DatasetDict = get_dataset("SHAPES", train=False)

num_holdout_samples = 10

unique_targets = set([x[1] for x in DataLoader(testset, batch_size=1)])

print(f"{unique_targets=}")

# NOTE: MNIST takes them based on labels, whereas we don't
HOLDOUT_DATA = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in HOLDOUT_DATA:
        HOLDOUT_DATA[target.item()] = data.squeeze()
        if len(HOLDOUT_DATA) == num_holdout_samples:
            break

HOLDOUT_DATA: Float[t.Tensor, "num_holdout_samples c=1 h=28 w=28"] = (
    t.stack([HOLDOUT_DATA[i] for i in range(num_holdout_samples)])
    .to(dtype=t.float, device=device)
    .unsqueeze(1)
)

display_data(HOLDOUT_DATA, nrows=2, title="MNIST holdout data")


# %%

## Note!
#
# For the rest of this section (not including the bonus), we'll assume we're working
# with the MNIST dataset rather than Celeb-A.

# %%

# add a dummy batch dimension
example_batch_mnist = einops.rearrange(HOLDOUT_DATA[0], "c h w -> 1 c h w")

print(example_batch_mnist.shape)

display_data(example_batch_mnist, nrows=1, title="example_batch_mnist")

# %%

import torch.nn as nn
import torch.nn.functional as F

import collections

from einops.layers.torch import Rearrange


class Autoencoder(nn.Module):

    def __init__(self, latent_dim_size: int, hidden_dim_size: int) -> None:
        super().__init__()

        if not (latent_dim_size < hidden_dim_size):
            raise ValueError(
                f"{latent_dim_size=} must be smaller " f"than {hidden_dim_size=}"
            )

        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

        # - Your encoder should consist of
        #   - two convolutional blocks (i.e. convolution plus ReLU)
        #   - followed by two fully connected linear layers with a ReLU in between them
        #   - Both convolutions will have kernel size 4, stride 2, padding 1
        #      - (recall this halves the size of the image).
        #   - We'll have 16 and 32 output channels respectively.
        #

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
            # go from the hidden dimension to the latent dimension (bottleneck)
            #
            # NOTE: THERE'S NO RELU, SO THIS IS JUST AN ENCODING
            nn.Linear(
                in_features=hidden_dim_size,
                out_features=latent_dim_size,
            ),
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
    def forward(self, x: Float[t.Tensor, "b c h w"]) -> Float[t.Tensor, "b c h w"]:

        encoded: Float[t.Tensor, "b d_latent"] = self.encoder(x)
        decoded: Float[t.Tensor, "b c h w"] = self.decoder(encoded)

        return decoded


model = Autoencoder(latent_dim_size=2, hidden_dim_size=4)

model.to(device)

print(model)

result = model.forward(example_batch_mnist)

display_data(result, nrows=1, title="example_batch_mnist")

# %%


@dataclass
class AutoencoderArgs:
    latent_dim_size: int = 16
    hidden_dim_size: int = 64
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: Tuple[float] = (0.9, 0.999)
    seconds_between_eval: int = 5
    wandb_project: Optional[str] = "day5-ae-shapes"
    wandb_name: Optional[str] = None


Loss = Float[t.Tensor, ""]


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs) -> None:

        self.args = args

        self.trainset = get_dataset(args.dataset)

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.model = Autoencoder(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)

        # TODO(bschoen): AdamW?
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=args.betas,
        )

        self.step = 0

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def training_step(self, img: Float[t.Tensor, "b c h w"]) -> Loss:
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        """

        self.optimizer.zero_grad()

        reconstructed_img: Float[t.Tensor, "b c h w"] = self.model(img)

        # calculate loss on the reconstructed image
        loss = F.mse_loss(img, reconstructed_img)

        # propagate loss backwards
        loss.backward()

        # step the optimizer
        self.optimizer.step()

        # return the loss so that we can log it
        return loss

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Evaluates model on holdout data, logs to weights & biases.
        """

        # TODO(bschoen): Do we need to do anything else when we have inference mode wrapper?
        self.model.eval()

        print(f"[{self.step=}] Evaluating on holdout data...")
        reconstructed_holdout = self.model(HOLDOUT_DATA)

        loss = F.mse_loss(HOLDOUT_DATA, reconstructed_holdout)

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
                "holdout_images": reshape_for_wandb(HOLDOUT_DATA),
                "reconstructed_holdout_images": reshape_for_wandb(
                    reconstructed_holdout
                ),
            },
            step=self.step,
        )

        # put model back in train mode
        self.model.train()

        wandb.log({"val_loss": loss.item()}, step=self.step)

        print(f"[{self.step=}] val_loss: {loss.item():.6f}")

    def train(self) -> None:
        """
        Performs a full training run, logging to wandb.
        """

        last_log_time = time.time()

        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        # wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for i, (img, label) in enumerate(
                progress_bar
            ):  # remember that label is not used

                img = img.to(device)

                loss = self.training_step(img)

                wandb.log(dict(loss=loss), step=self.step)

                # Evaluate model on the same holdout data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

                # Update `step`, which is used for wandb reporting
                self.step += img.shape[0]

                # Update progress bar
                progress_bar.set_description(
                    f"{epoch=}, {loss=:.4f}, examples_seen={self.step}"
                )

        wandb.finish()


args = AutoencoderArgs()
trainer = AutoencoderTrainer(args)
trainer.train()

# %%


@t.inference_mode()
def visualise_output(
    model: Autoencoder,
    n_points: int = 11,
    interpolation_range: Tuple[float, float] = (-3, 3),
) -> None:
    """
    Visualizes the output of the decoder, along the first two latent dims.
    """
    # Constructing latent dim data by making two of the dimensions vary indep in the interpolation range
    grid_latent = t.zeros(n_points**2, model.latent_dim_size).to(device)
    x = t.linspace(*interpolation_range, n_points).to(device)
    grid_latent[:, 0] = einops.repeat(x, "dim1 -> (dim1 dim2)", dim2=n_points)
    grid_latent[:, 1] = einops.repeat(x, "dim2 -> (dim1 dim2)", dim1=n_points)

    # Pass through decoder
    output = model.decoder(grid_latent).cpu().numpy()

    # Normalize & truncate, then unflatten back into a grid shape
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = einops.rearrange(
        output_truncated,
        "(dim1 dim2) 1 height width -> (dim1 height) (dim2 width)",
        dim1=n_points,
    )

    # Display the results
    tickargs = dict(
        tickmode="array",
        tickvals=list(range(14, 14 + 28 * n_points, 28)),
        ticktext=[f"{i:.2f}" for i in x],
    )
    px.imshow(
        output_single_image,
        color_continuous_scale="greys_r",
        title="Decoder output from varying first principal components of latent space",
    ).update_layout(
        xaxis=dict(title_text="dim1", **tickargs),
        yaxis=dict(title_text="dim2", **tickargs),
    ).show()


visualise_output(trainer.model)

# %%


@t.inference_mode()
def visualise_input(
    model: Autoencoder,
    dataset: Dataset,
) -> None:
    """
    Visualises (in the form of a scatter plot) the input data in the latent space, along the first two dims.
    """
    # First get the model images' latent vectors, along first 2 dims
    imgs = t.stack([batch for batch, label in dataset]).to(device)

    latent_vectors = model.encoder(imgs)

    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors[0]  # useful for VAEs later

    latent_vectors = latent_vectors[:, :2].cpu().numpy()

    labels = [str(label) for img, label in dataset]

    # Make a dataframe for scatter (px.scatter is more convenient to use when supplied with a dataframe)
    df = pd.DataFrame(
        {"dim1": latent_vectors[:, 0], "dim2": latent_vectors[:, 1], "label": labels}
    )
    df = df.sort_values(by="label")
    fig = px.scatter(df, x="dim1", y="dim2", color="label")
    fig.update_layout(
        height=700,
        width=700,
        title="Scatter plot of latent space dims",
        legend_title="Digit",
    )
    data_range = df["dim1"].max() - df["dim1"].min()

    # Add images to the scatter plot (optional)
    output_on_data_to_plot = model.encoder(HOLDOUT_DATA.to(device))[:, :2].cpu()

    if output_on_data_to_plot.ndim == 3:
        output_on_data_to_plot = output_on_data_to_plot[0]  # useful for VAEs; see later

    data_translated = (HOLDOUT_DATA.cpu().numpy() * 0.3081) + 0.1307

    data_translated = (255 * data_translated).astype(np.uint8).squeeze()

    for i in range(10):

        x, y = output_on_data_to_plot[i]

        fig.add_layout_image(
            source=Image.fromarray(data_translated[i]).convert("L"),
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=data_range / 15,
            sizey=data_range / 15,
        )

    fig.show()


small_dataset = Subset(get_dataset("SHAPES"), indices=range(0, 5000))
visualise_input(trainer.model, small_dataset)


# %%


# regularization := add a loss term

# basically add a loss

# For instance, with autoencoders there is no reason why we should expect
#   the linear interpolation between two points in the latent space to
#   have meaningful decodings.
#
# The decoder output will change continuously as we continuously vary the latent vector
# but that's about all we can say about it.
#
# However, if we use a variational autoencoder, we don't have this problem.
#  - The output of a linear interpolation between the cluster of 2s and cluster of 7s
#    will be "a symbol which pattern-matches to the family of MNIST digits, but has
#    equal probability to be interpreted as a 7, and this is indeed what we find.
#


# Intuitively, we can think of this as follows:
#  - when there is randomness in the process that generates the output,
#  - there is also randomness in the derivative of the output wrt the input
#    - so we can get a value for the derivative by sampling from this random distribution
#      - If we average over enough samples, this will give us a valid gradient for training.

# In sparse autoencoders, KL divergence is measuring:
#
#  - how different the activation distribution is from a desired sparse distribution
#
# In VAEs:
#
#  - regularizing the entire latent space distribution
#

# %%

# Beta SAE
#
# If each variable in the inferred latent representation is:
#  - only sensitive to one single generative factor
#  - relatively invariant to other factors
# we will say this representation is disentangled or factorized.
#
# One benefit that often comes with disentangled representation is
#  - good interpretability
#  - easy generalization to a variety of tasks.
#

# For example, a model trained on photos of human faces might capture the gender, skin color, hair color, hair length, emotion, whether wearing a pair of glasses and many other relatively independent factors in separate dimensions.
# Such a disentangled representation is very beneficial to facial image generation.

# β-VAE (Higgins et al., 2017) is a modification of Variational Autoencoder with a
# special emphasis to discover disentangled latent factors.
#
# Following the same incentive in VAE, we want to maximize the probability of generating
# real data, while keeping the distance between the real and estimated posterior
# distributions small (say, under a small constant):

# When β=1, it is same as VAE.
# When β>1, it applies a stronger constraint on the latent bottleneck and limits the representation capacity.
#
# For some conditionally independent generative factors, keeping them disentangled is the most efficient representation.
# Therefore a higher β encourages more efficient latent encoding and further encourages the disentanglement.
# Meanwhile, a higher β may create a trade-off between reconstruction quality and the extent of disentanglement.

# %%

Mean = Float[t.Tensor, "b d_latent"]
#
# note: we use `logsigma` instead of `sigma` solely because it's more numerically stable
#       - this seems to be a general pattern when we see log
#
LogStdDev = Float[t.Tensor, "b d_latent"]


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
        x: Float[t.Tensor, "b c h w"],
    ) -> tuple[Float[t.Tensor, "b d_latent"], Mean, LogStdDev]:
        """
        Passes `x` through the encoder. Returns the mean and log std dev of the latent vector,
        as well as the latent vector itself. This function can be used in `forward`, but also
        used on its own to generate samples for evaluation.
        """

        encoded_hidden: Float[t.Tensor, "b d_hidden"] = self.encoder(x)

        mu: Float[t.Tensor, "b d_latent"] = self.mu_layer(encoded_hidden)
        logsigma: Float[t.Tensor, "b d_latent"] = self.logsigma_layer(encoded_hidden)

        # recover sigma
        sigma = logsigma.exp()

        # sample `epsilon` from normal distribution
        epsilon: Float[t.Tensor, "b d_latent"] = t.normal(
            mean=t.zeros_like(mu),
            std=t.ones_like(sigma),
        )

        # z := mu + sigma * epsilon
        # <latent> = <average> + <stddev> * <sampled>
        #
        # where these are elementwise operations (as this is a reparameterization)
        z: Float[t.Tensor, "b d_latent"] = mu + sigma * epsilon

        return (z, mu, logsigma)

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[t.Tensor, "b c h w"],
    ) -> tuple[Float[t.Tensor, "b c h w"], Mean, LogStdDev]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well
        as mu and logsigma.
        """

        z, mu, logsigma = self.sample_latent_vector(x)

        x_prime: Float[t.Tensor, "b c h w"] = self.decoder(z)

        return (x_prime, mu, logsigma)


model = VAE(latent_dim_size=2, hidden_dim_size=4)

model.to(device)

print(model)

result, mu, logsigma = model.forward(example_batch_mnist)

print(f"{mu=}")
print(f"{logsigma.exp()=}")

display_data(result, nrows=1, title="example_batch_mnist")

sample, mu, logsigma = model.sample_latent_vector(example_batch_mnist)

print(f"{sample=}")
print(f"{mu=}")
print(f"{logsigma.exp()=}")

# display_data(sample, nrows=1, title="sample")

# %%


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
    mu: Float[t.Tensor, "b d_latent"],
    logsigma: Float[t.Tensor, "b d_latent"],
) -> Float[t.Tensor, ""]:

    # simplifies down to

    sigma_squared = t.exp(2 * logsigma)
    mu_squared = mu**2

    kl_div = (((sigma_squared + mu_squared) - 1) / 2) - logsigma

    # average over batch and latent dim (same as MSE does in practice)
    #
    return kl_div.mean()


loss = compute_kl_divergence_against_normal_distribution(
    mu=mu,
    logsigma=logsigma,
)

# ex: tensor([[0.1338, 0.1937]]
print(f"Loss: {loss}")

# %%


@dataclass
class VAELoss:
    total_loss: Float[t.Tensor, "b"]
    reconstruction_loss: Float[t.Tensor, "b"]
    kl_divergence_loss: Float[t.Tensor, "b"]
    mu: Mean
    sigma: Float[t.Tensor, "b"]


def compute_vae_loss(
    input: Float[t.Tensor, "b c h w"],
    reconstructed_input: Float[t.Tensor, "b c h w"],
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


vae_loss = compute_vae_loss(
    input=example_batch_mnist,
    reconstructed_input=result,
    mu=mu,
    logsigma=logsigma,
    beta_kl=0.1,
)

print(vae_loss)

# %%


@dataclass
class VAEArgs:
    latent_dim_size: int = 16
    hidden_dim_size: int = 512
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-4
    betas: Tuple[float] = (0.9, 0.999)

    # how much to penalize KL divergence
    beta_kl: float = 0.1
    seconds_between_eval: int = 5
    wandb_project: Optional[str] = "day5-ae-shapes"
    wandb_name: Optional[str] = None


class VAETrainer:
    def __init__(self, args: VAEArgs) -> None:

        self.args = args

        self.trainset = get_dataset(args.dataset)

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)

        # TODO(bschoen): AdamW?
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=args.betas,
        )

        self.step = 0

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def training_step(self, img: Float[t.Tensor, "b c h w"]) -> VAELoss:
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

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Evaluates model on holdout data, logs to weights & biases.
        """

        # TODO(bschoen): Do we need to do anything else when we have inference mode wrapper?
        self.model.eval()

        print(f"[{self.step=}] Evaluating on holdout data...")
        reconstructed_holdout, mu, logsigma = self.model(HOLDOUT_DATA)

        vae_loss = compute_vae_loss(
            HOLDOUT_DATA,
            reconstructed_holdout,
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
                "holdout_images": reshape_for_wandb(HOLDOUT_DATA),
                "reconstructed_holdout_images": reshape_for_wandb(
                    reconstructed_holdout
                ),
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
        # wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for i, (img, label) in enumerate(
                progress_bar
            ):  # remember that label is not used

                img = img.to(device)

                vae_loss = self.training_step(img)

                # log vae loss
                self._log_vae_loss(prefix="train", vae_loss=vae_loss)

                # Evaluate model on the same holdout data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

                # Update `step`, which is used for wandb reporting
                self.step += img.shape[0]

                # Update progress bar
                progress_bar.set_description(
                    f"{epoch=}, {vae_loss.total_loss.item()=:.4f}, examples_seen={self.step}"
                )

        wandb.finish()


args = VAEArgs()
trainer = VAETrainer(args)
trainer.train()

# %%


@t.inference_mode()
def visualise_input_vae(
    model: VAE,
    dataset: Dataset,
) -> None:
    """
    Visualises (in the form of a scatter plot) the input data in the latent space, along the first two dims.
    """
    # First get the model images' latent vectors, along first 2 dims
    imgs = t.stack([batch for batch, label in dataset]).to(device)

    latent_vectors, mu, logsigma = model.sample_latent_vector(imgs)

    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors[0]  # useful for VAEs later

    latent_vectors = latent_vectors[:, :2].cpu().numpy()

    labels = [str(label) for img, label in dataset]

    # Make a dataframe for scatter (px.scatter is more convenient to use when supplied with a dataframe)
    df = pd.DataFrame(
        {"dim1": latent_vectors[:, 0], "dim2": latent_vectors[:, 1], "label": labels}
    )
    df = df.sort_values(by="label")
    fig = px.scatter(df, x="dim1", y="dim2", color="label")
    fig.update_layout(
        height=700,
        width=700,
        title="Scatter plot of latent space dims",
        legend_title="Digit",
    )
    data_range = df["dim1"].max() - df["dim1"].min()

    # Add images to the scatter plot (optional)
    output_on_data_to_plot = model.encoder(HOLDOUT_DATA.to(device))[:, :2].cpu()

    if output_on_data_to_plot.ndim == 3:
        output_on_data_to_plot = output_on_data_to_plot[0]  # useful for VAEs; see later

    data_translated = (HOLDOUT_DATA.cpu().numpy() * 0.3081) + 0.1307

    data_translated = (255 * data_translated).astype(np.uint8).squeeze()

    for i in range(10):

        x, y = output_on_data_to_plot[i]

        fig.add_layout_image(
            source=Image.fromarray(data_translated[i]).convert("L"),
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=data_range / 15,
            sizey=data_range / 15,
        )

    fig.show()


# okay so it will separate pretty well

# can't we just measure average L_norm at this point to see n-dimensional sparseness
small_dataset = Subset(get_dataset("SHAPES"), indices=range(0, 5000))
visualise_input_vae(trainer.model, small_dataset)

# %%

visualise_output(trainer.model)

# %%

# VQ-SAE => quantized

# %%

from sklearn.decomposition import PCA
from sklearn.manifold import (
    TSNE,
)  # Import TSNE for dimensionality reduction and visualization


def to_tensor(array: np.ndarray) -> t.Tensor:
    return t.from_numpy(array).float().to(device)


# imagine taking a "top down" view with most important 2D as far as variance explained
@dataclass
class PCAResults:
    # first N vectors of the principal components
    principal_components: Float[t.Tensor, "n_pca d_latent"]

    # Projected data: The original data transformed into the new coordinate system defined by the principal components
    # Each row represents a data point, each column represents its coordinates along a principal component
    pca_vectors: Float[t.Tensor, "b n_pca"]

    # explained variance of the principal components
    explained_variance_ratio: Float[t.Tensor, "n_pca"]


@t.inference_mode()
def get_pca_components(
    model: VAE,
    dataset: Dataset,
    n_components: int = 2,
) -> PCAResults:
    """
    Gets the first `n_components` principal components in latent space

    Supports any model with `encoder`.

    TODO(bschoen): Protocol for `encoder`

    """

    # Unpack the (small) dataset into a single batch
    imgs = t.stack([batch[0] for batch in dataset]).to(device)
    labels = t.tensor([batch[1] for batch in dataset])

    # Get the latent vectors by passing images through the encoder
    latent_vectors, mu, sigma = model.sample_latent_vector(imgs.to(device))

    latent_vectors = latent_vectors.cpu().numpy()

    # Perform PCA to find the principal components
    pca = PCA(n_components=n_components)
    principal_components: Float[np.ndarray, "b n_pca"] = pca.fit_transform(
        latent_vectors
    )
    pca_vectors: Float[np.ndarray, "n_pca d_latent"] = pca.components_

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Convert numpy arrays to PyTorch tensors and return
    return PCAResults(
        pca_vectors=to_tensor(pca_vectors),
        principal_components=to_tensor(principal_components),
        explained_variance_ratio=to_tensor(explained_variance_ratio),
    )


small_dataset = Subset(get_dataset("SHAPES"), indices=range(0, 5000))

pca_results = get_pca_components(
    model=trainer.model,
    dataset=small_dataset,
    n_components=2,
)

# Print PCA results in a readable format
print("PCA Results:")
print(f"Principal components shape: {pca_results.principal_components.shape}")
print(f"PCA vectors shape: {pca_results.pca_vectors.shape}")
print("\nExplained variance ratio:")
for i, ratio in enumerate(pca_results.explained_variance_ratio, 1):
    print(f"Component {i}: {ratio.item():.4f}")

print("\nFirst two PCA vectors:")
for i, vector in enumerate(pca_results.pca_vectors[:2], 1):
    print(f"Vector {i}: {vector[:5]}...")  # Print first 5 elements

# these are big
print("\nFirst few principal components:")
print(pca_results.principal_components[:5, :])

# %%

import matplotlib.pyplot as plt
from typing import List


@t.inference_mode()
def visualize_pca_results(
    model: VAE,
    dataset: Dataset,
    pca_results: PCAResults,
    n_components: int = 2,
    n_examples: int = 5,
) -> None:
    """
    Visualize PCA results with max and min activating examples for each principal component.

    Args:
        model: The trained VAE model
        dataset: The dataset used for PCA
        pca_results: PCA results from get_pca_components
        n_components: Number of principal components to visualize
        n_examples: Number of max/min activating examples to show for each component
    """
    # Get all images and their latent representations
    imgs = t.stack([batch[0] for batch in dataset]).to(device)
    latent_vectors, _, _ = model.sample_latent_vector(imgs)

    # Project latent vectors onto PCA components
    projected = t.matmul(latent_vectors, pca_results.pca_vectors.T)

    fig, axes = plt.subplots(
        n_components, 2 * n_examples + 1, figsize=(20, 3 * n_components)
    )

    for i in range(n_components):
        # Sort images by their projection onto this component
        sorted_indices = projected[:, i].argsort()

        # Plot the component vector
        axes[i, 0].bar(
            range(latent_vectors.shape[1]),
            pca_results.pca_vectors.cpu()[i],
        )
        axes[i, 0].set_title(f"PC {i+1}")
        axes[i, 0].set_xlabel("Latent dim")
        axes[i, 0].set_ylabel("Weight")

        # Plot min activating examples
        for j in range(n_examples):
            idx = sorted_indices[j]
            img = imgs[idx].cpu().squeeze()
            axes[i, j + 1].imshow(img.cpu(), cmap="gray")
            axes[i, j + 1].axis("off")
            if j == 0:
                axes[i, j + 1].set_title(f"Min {n_examples} examples")

        # Plot max activating examples
        for j in range(n_examples):
            idx = sorted_indices[-(j + 1)]
            img = imgs[idx].cpu().squeeze()
            axes[i, j + n_examples + 1].imshow(img.cpu(), cmap="gray")
            axes[i, j + n_examples + 1].axis("off")
            if j == 0:
                axes[i, j + n_examples + 1].set_title(f"Max {n_examples} examples")

    plt.tight_layout()
    plt.show()


# Visualize PCA results
visualize_pca_results(trainer.model, small_dataset, pca_results)

# %%%


@t.inference_mode()
def interpolate_latent_space(
    model: VAE,
    pca_results: PCAResults,
    n_steps: int = 10,
    range_multiplier: float = 3.0,
) -> None:
    """
    Interpolate along the first two principal components and visualize the results.

    Args:
        model: The trained VAE model
        pca_results: PCA results from get_pca_components
        n_steps: Number of steps for interpolation
        range_multiplier: Multiplier for the range of interpolation
    """
    # Create a grid of points in the 2D PCA space
    x = np.linspace(-range_multiplier, range_multiplier, n_steps)
    y = np.linspace(-range_multiplier, range_multiplier, n_steps)
    xx, yy = np.meshgrid(x, y)

    # Convert grid points to latent space
    grid_2d = (
        t.tensor(
            np.column_stack(
                [
                    xx.ravel(),
                    yy.ravel(),
                ]
            )
        )
        .float()
        .to(device)
    )
    latent_grid = t.matmul(grid_2d, pca_results.pca_vectors[:2])

    # Generate images from latent grid
    with t.no_grad():
        generated_imgs = model.decoder(latent_grid).cpu()

    # Plot the results
    fig, axes = plt.subplots(n_steps, n_steps, figsize=(15, 15))
    for i in range(n_steps):
        for j in range(n_steps):
            idx = i * n_steps + j
            img = generated_imgs[idx].squeeze()
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")

    plt.suptitle("Interpolation along first two principal components")
    plt.tight_layout()
    plt.show()


# Interpolate and visualize latent space
interpolate_latent_space(trainer.model, pca_results)

# %%


@t.inference_mode()
def interpolate_latent_space_tsne(
    model: VAE,
    dataset: Dataset,
    n_steps: int = 10,
    range_multiplier: float = 3.0,
    n_samples: int = 1000,
) -> None:
    """
    Interpolate along the first two t-SNE components and visualize the results.

    Args:
        model: The trained VAE model
        dataset: The dataset to sample from
        n_steps: Number of steps for interpolation
        range_multiplier: Multiplier for the range of interpolation
        n_samples: Number of samples to use for t-SNE
    """
    # Sample a subset of the dataset
    subset = t.utils.data.Subset(dataset, range(n_samples))
    dataloader = t.utils.data.DataLoader(subset, batch_size=n_samples)

    images, _ = next(iter(dataloader))
    images = images.to(device)

    # Get latent representations
    latent_vectors, _, _ = model.sample_latent_vector(images)
    latent_vectors = latent_vectors.cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_vectors)

    # Create a grid of points in the 2D t-SNE space
    x = np.linspace(tsne_results[:, 0].min(), tsne_results[:, 0].max(), n_steps)
    y = np.linspace(tsne_results[:, 1].min(), tsne_results[:, 1].max(), n_steps)
    xx, yy = np.meshgrid(x, y)

    # Convert grid points to latent space
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # Find nearest neighbors in t-SNE space
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(tsne_results)
    _, indices = nn.kneighbors(grid_2d)

    # Get corresponding latent vectors
    latent_grid = t.tensor(latent_vectors[indices.squeeze()]).float().to(device)

    # Generate images from latent grid
    with t.no_grad():
        generated_imgs = model.decoder(latent_grid).cpu()

    # Plot the results
    fig, axes = plt.subplots(n_steps, n_steps, figsize=(15, 15))
    for i in range(n_steps):
        for j in range(n_steps):
            idx = i * n_steps + j
            img = generated_imgs[idx].squeeze()
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")

    plt.suptitle("Interpolation along first two t-SNE components")
    plt.tight_layout()
    plt.show()


# Interpolate and visualize latent space using t-SNE
interpolate_latent_space_tsne(trainer.model, small_dataset)

# %%


@t.inference_mode()
def analyze_latent_distribution(model, dataset, n_samples=1000):
    # Sample a subset of the dataset
    subset = t.utils.data.Subset(dataset, range(n_samples))
    dataloader = t.utils.data.DataLoader(subset, batch_size=n_samples)

    images, _ = next(iter(dataloader))
    images = images.to(device)

    # Get latent representations
    _, mu, logsigma = model.sample_latent_vector(images)
    mu = mu.cpu().numpy()
    sigma = t.exp(logsigma).cpu().numpy()

    # Plot histograms of mu and sigma
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(mu.flatten(), bins=50)
    axes[0].set_title("Distribution of μ")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(sigma.flatten(), bins=50)
    axes[1].set_title("Distribution of σ")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


analyze_latent_distribution(trainer.model, small_dataset)

# %%

from sklearn.manifold import TSNE


@t.inference_mode()
def visualize_latent_space_tsne(model, dataset, n_samples=1000):
    # Sample a subset of the dataset
    subset = t.utils.data.Subset(dataset, range(n_samples))
    dataloader = t.utils.data.DataLoader(subset, batch_size=n_samples)

    images, labels = next(iter(dataloader))
    images = images.to(device)

    # Get latent representations
    z, _, _ = model.sample_latent_vector(images)
    z = z.cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(z)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("t-SNE of 64D Latent Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


visualize_latent_space_tsne(trainer.model, small_dataset)


# %%
@t.inference_mode()
def latent_correlation_analysis(model, dataset, n_samples=1000):
    # Sample a subset of the dataset
    subset = t.utils.data.Subset(dataset, range(n_samples))
    dataloader = t.utils.data.DataLoader(subset, batch_size=n_samples)

    images, labels = next(iter(dataloader))
    images = images.to(device)

    # Get latent representations
    z, _, _ = model.sample_latent_vector(images)
    z = z.cpu().numpy()

    # Compute correlations between latent dimensions and labels
    correlations = np.array([np.corrcoef(z[:, i], labels)[0, 1] for i in range(16)])

    # Plot correlations
    plt.figure(figsize=(15, 5))
    plt.bar(range(16), correlations)
    plt.title("Correlation between Latent Dimensions and Digit Labels")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Correlation Coefficient")
    plt.show()


latent_correlation_analysis(trainer.model, small_dataset)

# %%


def decoder_layer_activations(model, latent_vector, retain_grad=False):
    activations = []

    def hook_fn(module, input, output):
        # If retain_grad is True, we don't detach the output
        # This allows gradients to flow through if needed
        act = output if retain_grad else output.detach()
        activations.append(act)

    hooks = []
    for layer in model.decoder:
        if isinstance(layer, (nn.Linear, nn.ConvTranspose2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    model.decoder(latent_vector)

    for hook in hooks:
        hook.remove()

    return activations


@t.inference_mode()
def visualize_decoder_activations(model, latent_vector):
    activations = decoder_layer_activations(model, latent_vector)

    print(f"Visualizing decoder activations: {len(activations)=}")

    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))

    for i, act in enumerate(activations):

        if act.dim() == 4:  # For convolutional layers

            act = act.mean(dim=1)  # Average over channels

        elif act.dim() == 2:  # For linear layers

            # Reshape linear layer output to 40x40 square image
            if act.shape[1] == 1568:
                side_length = 40
                act = act.view(1, 1568)
                act = nn.functional.pad(act, (0, 32))  # Pad to 1600
                act = act.view(side_length, side_length)
            else:
                # Fallback to square reshaping or padding if not 1568
                print(f"Reshaping {act.shape}")
                num_elements = act.shape[1]
                side_length = int(num_elements**0.5)
                if side_length**2 == num_elements:
                    # If perfect square, reshape directly
                    act = act.view(side_length, side_length)
                else:
                    # If not perfect square, pad to next perfect square
                    next_square = (side_length + 1) ** 2
                    act = nn.functional.pad(act, (0, next_square - num_elements))
                    act = act.view(side_length + 1, side_length + 1)

        axes[i].imshow(act.cpu().squeeze(), cmap="Spectral")
        axes[i].set_title(f"Layer {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
latent_vector = t.randn(1, 16, device=device)
visualize_decoder_activations(trainer.model, latent_vector)

# %%


@t.inference_mode()
def encoder_layer_activations(model, input_image):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    hooks = []
    for layer in model.encoder:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Add hooks for mu and logsigma layers
    hooks.append(model.mu_layer.register_forward_hook(hook_fn))
    hooks.append(model.logsigma_layer.register_forward_hook(hook_fn))

    model.encoder(input_image)

    for hook in hooks:
        hook.remove()

    return activations


@t.inference_mode()
def visualize_encoder_activations(model, input_image):
    activations = encoder_layer_activations(model, input_image)

    print(f"Visualizing encoder activations: {len(activations)=}")

    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))

    for i, act in enumerate(activations):
        if act.dim() == 4:  # For convolutional layers
            act = act.mean(dim=1)  # Average over channels
        elif act.dim() == 2:  # For linear layers
            print(f"Reshaping {act.shape}")
            num_elements = act.shape[1]
            side_length = int(num_elements**0.5)
            if side_length**2 == num_elements:
                # If perfect square, reshape directly
                act = act.view(side_length, side_length)
            else:
                # If not perfect square, pad to next perfect square
                next_square = (side_length + 1) ** 2
                act = nn.functional.pad(act, (0, next_square - num_elements))
                act = act.view(side_length + 1, side_length + 1)

        axes[i].imshow(act.cpu().squeeze(), cmap="viridis")
        axes[i].set_title(f"Layer {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


visualize_encoder_activations(trainer.model, example_batch_mnist)

# %%


@t.inference_mode()
def visualize_full_vae_activations(model, input_image):
    # Encoder activations
    encoder_activations = encoder_layer_activations(model, input_image)

    # Get latent representation
    z, _, _ = model.sample_latent_vector(input_image)

    # Decoder activations
    decoder_activations = decoder_layer_activations(model, z)

    all_activations = encoder_activations + [z] + decoder_activations

    print(f"Visualizing full VAE activations: {len(all_activations)=}")

    fig, axes = plt.subplots(1, len(all_activations), figsize=(25, 4))

    for i, act in enumerate(all_activations):
        if act.dim() == 4:  # For convolutional layers
            act = act.mean(dim=1)  # Average over channels
        elif act.dim() == 2:  # For linear layers
            print(f"Reshaping {act.shape}")
            num_elements = act.shape[1]
            side_length = int(num_elements**0.5)
            if side_length**2 == num_elements:
                # If perfect square, reshape directly
                act = act.view(-1, side_length, side_length)
            else:
                # If not perfect square, pad to next perfect square
                next_square = (side_length + 1) ** 2
                act = nn.functional.pad(act, (0, next_square - num_elements))
                act = act.view(-1, side_length + 1, side_length + 1)

        axes[i].imshow(act.cpu().squeeze(), cmap="viridis")
        axes[i].set_title(f"Layer {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
visualize_full_vae_activations(trainer.model, example_batch_mnist)

# %%


def optimize_latent_for_neuron_old(
    model, layer_idx, neuron_idx, n_steps=10000, lr=0.01
):
    print(f"\nOptimizing latent vector for layer {layer_idx}, neuron {neuron_idx}")
    print(
        "This process aims to find the input that maximally activates a specific neuron"
    )

    z = t.randn(1, 16, requires_grad=True, device=device)
    optimizer = t.optim.Adam([z], lr=lr)

    model.eval()  # Set model to evaluation mode

    for step in range(n_steps):
        optimizer.zero_grad()

        with t.enable_grad():  # Enable gradient computation
            activations = decoder_layer_activations(model, z, retain_grad=True)
            target_activation = activations[layer_idx][0, neuron_idx]
            loss = -target_activation.mean()  # Negative sign to maximize activation

        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}/{n_steps}, Activation: {-loss.item():.4f}")

    print(f"Optimization complete. Final activation: {-loss.item():.4f}")
    return z.detach()


def optimize_latent_for_neuron(
    model,
    layer_idx,
    neuron_idx,
    n_steps=1000,
    lr=0.1,
    n_tries=3,
):
    print(f"\nOptimizing latent vector for layer {layer_idx}, neuron {neuron_idx}")
    print(
        "This process aims to find the input that maximally activates a specific neuron"
    )

    best_z = None
    best_activation = float("-inf")

    for try_num in range(n_tries):
        print(f"\n[{layer_idx=}][{neuron_idx=}]Try {try_num + 1}/{n_tries}")

        z = t.randn(1, 16, requires_grad=True, device=device)

        optimizer = t.optim.Adam([z], lr=lr)

        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=100,
            factor=0.5,
        )

        model.eval()
        best_try_activation = float("-inf")
        steps_without_improvement = 0

        for step in range(n_steps):
            optimizer.zero_grad()

            with t.enable_grad():
                activations = decoder_layer_activations(model, z, retain_grad=True)
                target_activation = activations[layer_idx][0, neuron_idx]
                regularization = 0.01 * t.norm(z)  # L2 regularization
                loss = -target_activation.mean() + regularization

            loss.backward()
            t.nn.utils.clip_grad_norm_([z], max_norm=1.0)  # Gradient clipping
            optimizer.step()

            current_activation = -loss.item()
            scheduler.step(current_activation)

            if current_activation > best_try_activation:
                best_try_activation = current_activation
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if step % 100 == 0:
                print(f"Step {step}/{n_steps}, Activation: {current_activation:.4f}")

            if steps_without_improvement >= 100:  # Early stopping
                print(f"Early stopping at step {step}")
                break

        if best_try_activation > best_activation:
            best_activation = best_try_activation
            best_z = z.detach().clone()

    print(f"Optimization complete. Best activation: {best_activation:.4f}")
    return best_z


def visualize_neuron_features(model, layer_idx, n_neurons: int = 16):
    # Get the total number of neurons in the specified layer
    with t.no_grad():
        dummy_input = t.randn(1, 16).to(device)
        activations = decoder_layer_activations(model, dummy_input)

        print(f"{len(activations)=}")

        total_neurons = activations[layer_idx].shape[1]

    print(
        f"\nVisualizing features detected by {n_neurons} out of "
        f"{total_neurons} neurons in layer {layer_idx}"
    )
    print(
        "This helps us understand what patterns or features "
        "each neuron is sensitive to"
    )

    fig, axes = plt.subplots(1, n_neurons, figsize=(4 * n_neurons, 4))
    for i in range(n_neurons):
        print(f"\nProcessing neuron {i} (neuron {i}/{total_neurons} in the layer)")
        z = optimize_latent_for_neuron(model, layer_idx, i)
        with t.no_grad():  # Disable gradient computation for inference
            img = model.decoder(z).cpu().squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Neuron {i}/{total_neurons}")
        axes[i].axis("off")
        print(f"Image for neuron {i}/{total_neurons} generated and plotted")

    plt.tight_layout()
    print("\nDisplaying the visualization...")
    plt.show()
    print(
        f"Visualization complete. Each image shows the input that maximally activates a specific neuron out of {total_neurons} in the layer."
    )


visualize_neuron_features(trainer.model, layer_idx=-2)

# %%

print(trainer.model.decoder)

# %%
