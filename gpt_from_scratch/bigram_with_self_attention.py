#!/usr/bin/env python
# coding: utf-8

import pathlib
from typing import Unpack
import math

import file_utils
import vocab_utils

import torch
import torch.nn as nn
from torch.nn import functional as F

# imported for typechecking
#
# note: can't easily alias via jaxtyping annotations, as it's a string literal and
#       likely plays weirdly with typing.Annotation to forward a payload
# note: torchtyping is deprecated in favor of jaxtyping, as torchtyping doesn't have mypy integration
#
# note: jaxtyping does support prepending
#
#   Image = Float[Array, "channels height width"]
#   BatchImage = Float[Image, "batch"]
#
#    -->
#
#   BatchImage = Float[Array, "batch channels height width"]
#
# so we can compose aliases
#
from torch import Tensor
import jaxtyping
from jaxtyping import jaxtyped, Float32, Int64
from typeguard import typechecked as typechecker

# now we still want to batch
# we seed it so it's always the same
torch.manual_seed(1337)

# type aliases
type Block = Int64[Tensor, "block_size"]

# equivalent to `Int64[Tensor, "batch_size block_size"]`
type BatchedBlocks = Int64[Block, "batch_size"]


# TODO(bschoen): Understanding dropout
"""
Adding dropout

[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

Essentially causes the model to train a handful of ensemble networks

This seems super hand wavy.

"""


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(
        self,
        d_model: int,
        n_embed: int,
        block_size: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.n_embed = n_embed
        self.d_model = d_model

        self.key = nn.Linear(n_embed, d_model, bias=False)
        self.query = nn.Linear(n_embed, d_model, bias=False)
        self.value = nn.Linear(n_embed, d_model, bias=False)

        # TODO(bschoen): Why?
        tril: Float32[Tensor, "block_size block_size"] = torch.ones(
            block_size, block_size
        )
        tril = torch.tril(tril)
        self.register_buffer("tril", tril)

        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float32[Tensor, "batch_size block_size n_embed"],
    ) -> Float32[Tensor, "batch_size block_size d_model"]:

        B, T, C = x.shape

        k: Float32[Tensor, "batch_size block_size d_model"] = self.key(x)
        q: Float32[Tensor, "batch_size block_size d_model"] = self.query(x)

        # compute attention scores ("affinities"), this is our QK matrix
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights: Float32[Tensor, "batch_size block_size block_size"] = q @ k.transpose(
            -2, -1
        )

        # normalize the attention scores
        weights = weights / math.sqrt(self.n_embed)

        # masked attention so can only see previous tokens
        # note: `T` is used instead of `block_size` to allow for sequences shorter than `block_size`
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # (B, T, T)
        weights = F.softmax(weights, dim=-1)

        # TODO(bschoen): Why apply dropout here? Is it always right before it actually gets used?
        weights = self.dropout(weights)

        # perform the weighted aggregation of the values
        v: Float32[Tensor, "batch_size block_size d_model"] = self.value(x)

        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel, then concatenating the results

    "these tokens have a lot to talk about"

    Helpful to think of these as multiple independent channels.

    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        n_embed: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_embed = n_embed

        if n_heads * d_model != n_embed:
            raise ValueError(
                f"{n_heads * d_model=} ({n_heads=}, {d_model=}) != {n_embed=}. "
                f"Must be able to stack the outputs of {n_heads=} of dimension {d_model=} "
                f"into an output that's back in the embedding space {n_embed=}"
            )

        self.heads = nn.ModuleList(
            [
                Head(
                    d_model=d_model,
                    n_embed=n_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_heads)
            ]
        )

        # TODO(bschoen): Don't we want this to go from `d_model` to `n_embed`?
        self.proj = nn.Linear(n_embed, n_embed)

        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float32[Tensor, "batch_size block_size n_embed"],
    ) -> Float32[Tensor, "batch_size block_size n_embed"]:

        head_outputs: list[Float32[Tensor, "batch_size block_size d_model"]] = [
            head.forward(x) for head in self.heads
        ]

        # assertion just for readability, we already check this during construction
        assert len(head_outputs) * self.d_model == self.n_embed

        # concatenate them over the last dimension
        out: Float32[Tensor, "batch_size block_size n_embed"] = torch.cat(
            head_outputs,
            dim=-1,
        )

        # TODO(bschoen): Why this projection?
        out = self.proj(out)

        out = self.dropout(out)

        return out


class FeedFoward(nn.Module):
    """
    a simple linear layer followed by a non-linearity

    Want to add computation into the network

    Want that computation to be able to operate at a per token level.

    "Tokens looked at each other, but didn't really have a lot of time to 'think on' (a lot of compute)
     what they found."

    Note:
     - This is on a per token level
     - Self-attention is the communication
     - Now they need to think on that data individually

    """

    # note: this just comes directly from attention is all you need paper
    # TODO(bschoen): ???
    INNER_DIMENSIONALITY_FACTOR = 4

    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()

        d_model = self.INNER_DIMENSIONALITY_FACTOR * n_embed

        self.net = nn.Sequential(
            nn.Linear(n_embed, d_model),
            nn.ReLU(),
        )

        self.proj = nn.Linear(d_model, n_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float32[Tensor, "batch_size block_size n_embed"],
    ) -> Float32[Tensor, "batch_size block_size n_embed"]:

        out_model: Float32[Tensor, "batch_size block_size d_model"] = self.net.forward(
            x
        )

        out: Float32[Tensor, "batch_size block_size n_embed"] = self.proj.forward(
            out_model
        )

        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation

    Now that these are getting deep, they're getting harder to optimize

    (1) Let's use residual connections:

    [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Initialize residual blocks to near zero, and let them over time represent
    how much should contribute

    So:

        x = self.sa_heads(x)
        x = self.ffwd(x)

    Becomes:

        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)

    # TODO(bschoen): Why do we need this?

    Now both `MultiHeadAttention` and `FeedFoward` need an additional
    projection `proj` (nn.Linear(n_embed, n_embed))

    (2) Now let's add in layer normalization

    [Layer Normalization](https://arxiv.org/abs/1607.06450)

    Note: Layer norm has it's own `gamma` and `beta` parameters, which are
          learnable.

          def layer_norm(x):

            x_normalized = (x - torch.mean(x)) / math.sqrt(torch.var(x) + epsilon)

            out = x_normalized * gamma + beta

            return out

    Note: In the original `attention` paper, this was applied after attention / feed forward,
          now it's common to apply them before

    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        n_embed: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # note: normalization happening over embedding space
        self.ln1 = nn.LayerNorm(n_embed)

        # self attention head
        # ex: 4 heads, 8-dimensional self attention
        self.sa_heads = MultiHeadAttention(
            n_heads=n_heads,
            n_embed=n_embed,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embed)

        self.ffwd = FeedFoward(n_embed=n_embed, dropout=dropout)

    def forward(
        self,
        x: Float32[Tensor, "batch_size block_size n_embed"],
    ) -> Float32[Tensor, "batch_size block_size n_embed"]:

        # apply self attention
        x = self.ln1(x)
        x = x + self.sa_heads(x)

        # feed forward layer to "think on" the results of the self attention
        x = self.ln2(x)
        x = x + self.ffwd(x)

        return x


# Adding in token embeddings means we need a linear layer
# (to go from token embeddings to logits, basically to undo the linear embedding layer)
# Is this just an unembedding layer?
#
# We're basically add a bunch of stuff onto just bigram *first*
#
class BigramWithSelfAttentionLanguageModel(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_transformer_blocks: int,
        block_size: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.block_size = block_size

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # add position embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # sequential decoder only transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_heads=n_heads,
                    n_embed=n_embed,
                    d_model=n_embed // n_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_transformer_blocks)
            ],
            nn.LayerNorm(n_embed),
        )

        # map from token embeddings back to logits
        self.lm_head = nn.Linear(n_embed, vocab_size)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        idx: BatchedBlocks,
        targets: BatchedBlocks | None = None,
    ) -> tuple[
        Float32[Tensor, "batch_size block_size vocab_size"]
        | Float32[Tensor, "batch_size*block_size vocab_size"],
        Float32[Tensor, ""] | None,
    ]:

        # idx and targets are both (B,T) tensor of integers
        token_embeddings: Float32[Tensor, "batch_size block_size n_embed"] = (
            self.token_embedding_table(idx)
        )

        # ex: [0, 1, 2, 3, ...]
        pos_indices = torch.arange(idx.shape[1], device=idx.device)
        pos_embeddings: Float32[Tensor, "batch_size block_size n_embed"] = (
            self.position_embedding_table(pos_indices)
        )

        # concat
        x = token_embeddings + pos_embeddings

        x = self.transformer_blocks(x)

        # takes them back from token embedding space to logits
        logits: Float32[Tensor, "batch_size block_size vocab_size"] = self.lm_head(x)

        # if no targets, nothing to calculate
        if targets is None:
            return logits, None

        B, T, C = logits.shape

        # strech them out into 1d sequence, just because of quirks of what pytorch expects
        # for the cross_entropy calculation
        reshaped_logits: Float32[Tensor, "batch_size*block_size vocab_size"] = (
            logits.view(B * T, C)
        )

        reshaped_targets: Float32[Tensor, "batch_size*block_size"] = targets.view(B * T)

        loss: Float32[Tensor, ""] = F.cross_entropy(reshaped_logits, reshaped_targets)

        return logits, loss

    @jaxtyped(typechecker=typechecker)
    def generate(
        self,
        idx: BatchedBlocks,
        max_new_tokens: int,
    ) -> BatchedBlocks:

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop `idx` to the last `block_size` tokens, since that's
            # all our positional embedding supports
            idx_cond = idx[:, -self.block_size :]

            # get the predictions
            logits, loss = self.forward(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            # essentially gets the highest probability logit
            probs: Float32[Tensor, "batch_size vocab_size"] = F.softmax(
                logits, dim=-1
            )  # (B, C)

            # sample from the distribution
            idx_next: Float32[Tensor, "batch_size 1"] = torch.multinomial(
                probs, num_samples=1
            )  # (B, 1)

            # append sampled index to the running sequence
            # and move `idx` forward
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# want to average loss over a few batches
# tells pytorch doesn't have to store intermediate values
@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: Int64[Tensor, "num_samples"],
    num_batches_to_eval: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> float:

    # set to eval mode
    model.eval()

    # compute loss over `eval_interval` batches, then average it
    losses = torch.zeros(num_batches_to_eval)

    for i in range(num_batches_to_eval):
        # sample a batch of data
        xb, yb = get_batch(
            data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )

        # compute loss
        _, loss = model(xb, yb)

        losses[i] = loss

    # convert back to train mode
    model.train()

    # return average loss
    return losses.mean().item()


# note: usually want to stack into a batch
@jaxtyped(typechecker=typechecker)
def get_batch(
    data: Int64[Tensor, "num_samples"],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[BatchedBlocks, BatchedBlocks]:
    """Generate a small batch of data of inputs x and targets y."""

    # Generate 'batch_size' random indices. Each index is the start of a sequence.
    # The upper bound (len(data) - block_size) ensures we have enough room for a full sequence.
    max_batch_start_index = len(data) - block_size

    # choose `batch_size` random starting indices for where to start each batch
    batch_start_indices: Int64[Tensor, "batch_size"] = torch.randint(
        max_batch_start_index, (batch_size,)
    )

    # For each random start index, extract a sequence of length 'block_size'.
    x_blocks: list[Int64[Tensor, "block_size"]] = [
        data[i : i + block_size] for i in batch_start_indices
    ]

    # Similar to x, but shifted one position to the right (next-token prediction).
    # This creates the targets for each input sequence.
    y_blocks: list[Int64[Tensor, "block_size"]] = [
        data[i + 1 : i + block_size + 1] for i in batch_start_indices
    ]

    # Stack these sequences into a single tensor of shape (batch_size, block_size).
    x_batch: BatchedBlocks = torch.stack(x_blocks)
    y_batch: BatchedBlocks = torch.stack(y_blocks)

    # move to device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    return x_batch, y_batch


@torch.no_grad()
def generate_and_decode_text(
    m: nn.Module,
    vocab: vocab_utils.Vocabulary,
    device: torch.device,
    max_new_tokens: int = 500,
) -> str:

    # set to eval mode
    m.eval()

    # note: we're using batch size
    generated_text_batch = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_text_batch = m.generate(
        idx=generated_text_batch,
        max_new_tokens=max_new_tokens,
    )

    # check that we only generated one batch (since we only had one batch)
    assert len(generated_text_batch) == 1

    generated_text = generated_text_batch[0]

    # decode
    decoded_generated_text = vocab.decode(generated_text.tolist())

    # convert back to train mode
    m.train()

    return decoded_generated_text


# -- now we get to the part specific for our problem

# how many independent sequences will we process in parallel?
BATCH_SIZE = 4

# what is the maximum context length for predictions?
BLOCK_SIZE = 8

# how many batches to look at when running `evaluate_loss`
# 300 comes from karpathy
# note: this is likely why our numbers are a bit different
NUM_BATCHES_TO_EVAL = 100

# how often to evaluate the model
EVALUATE_EVERY_N_STEPS = 1000

MAX_STEPS = 10000

N_EMBED = 32

LEARNING_RATE = 1e-3

# number of self attention heads
N_HEADS = 4

DROPOUT = 0.1

N_TRANSFORMER_BLOCKS = 4


# Results (with 9000 steps, not 10,000 only due to printing error)
#
# note:
# - without self attention: 2.5
# - with single head self attention: 2.43
# - with multi-head (4) self attention: 2.35
# - ... + feed forward (in the attention block): 2.23
# - ... + 4 transformer blocks: 2.62 (worse!)
# - ... + residual connections: 2.16 (better than even before!)
# - ... + ff inner dim factor (4): 2.11
# - ... + layer norm: 2.14
# - ... + layer norm + final layer norm after transformer blocks: 2.14
# - ... + dropout:
def main() -> None:

    # note: does this cover the attention head too? presumably every sub module
    assert torch.backends.mps.is_available()
    device = torch.device("mps")

    # load tinyshakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    input_filepath = file_utils.download_file_from_url(url)

    # Read all text from the input file
    input_text = input_filepath.read_text()

    vocab = vocab_utils.Vocabulary(input_text)

    # let's now encode the entire text dataset and store it into a torch.Tensor
    encoded_input_text: Int64[Tensor, "num_samples"] = torch.tensor(
        vocab.encode(input_text),
        dtype=torch.long,
        device=device,
    )

    # Let's now split up the data into train and validation sets

    # first 90% will be train, rest val
    train_val_ratio = 0.9

    n = int(train_val_ratio * len(encoded_input_text))

    train_data: Int64[Tensor, "num_samples"] = encoded_input_text[:n]
    val_data: Int64[Tensor, "num_samples"] = encoded_input_text[n:]

    # note: train loss getting lower than val means we're overfitting
    print(f"Splitting {len(encoded_input_text)} input tokens into")
    print(f" - train: {len(train_data)}")
    print(f" - val: {len(val_data)}")

    m = BigramWithSelfAttentionLanguageModel(
        vocab_size=len(vocab.unique_elements),
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_transformer_blocks=N_TRANSFORMER_BLOCKS,
        dropout=DROPOUT,
    )
    m.to(device)

    print("Output before optimizing:\n---")
    print(generate_and_decode_text(m, vocab, device))
    print("---")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    # increase number of steps for good results...
    print(
        f"Running for {MAX_STEPS}, evaluating every {EVALUATE_EVERY_N_STEPS} steps..."
    )

    def evaluate_and_show_loss(step: int) -> None:
        # evaluate the loss
        loss_by_dataset_name: dict[str, float] = {}
        for name, data in [("train", train_data), ("val", val_data)]:
            loss_by_dataset_name[name] = estimate_loss(
                m,
                data,
                num_batches_to_eval=NUM_BATCHES_TO_EVAL,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                device=device,
            )

        train_loss = loss_by_dataset_name["train"]
        val_loss = loss_by_dataset_name["val"]

        print(f"Step {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print(f"Sampling at {step}:\n---")
        print(print(generate_and_decode_text(m, vocab, device, max_new_tokens=100)))
        print("---")

    # note: Karpathy is printing at the end
    for step in range(MAX_STEPS):

        if step % EVALUATE_EVERY_N_STEPS == 0:
            evaluate_and_show_loss(step=step)

        # sample a batch of data
        xb, yb = get_batch(
            train_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )

        # evaluate the loss
        logits, loss = m(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

    print("Final loss:")
    evaluate_and_show_loss(step=MAX_STEPS)

    print("Output after optimizing:\n---")
    print(print(generate_and_decode_text(m, vocab, device)))
    print("---")


if __name__ == "__main__":
    main()
