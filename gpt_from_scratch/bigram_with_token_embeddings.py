#!/usr/bin/env python
# coding: utf-8

import pathlib
from typing import Unpack

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


# Adding in token embeddings means we need a linear layer
# (to go from token embeddings to logits, basically to undo the linear embedding layer)
# Is this just an unembedding layer?
#
# We're basically add a bunch of stuff onto just bigram *first*
#
class BigramEnhancedLanguageModel(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        block_size: int,
    ) -> None:

        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # add position embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

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

            # get the predictions
            logits, loss = self.forward(idx)

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


def generate_and_decode_text(
    m: nn.Module,
    vocab: vocab_utils.Vocabulary,
    device: torch.device,
    max_new_tokens: int = 100,
) -> str:

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

    return decoded_generated_text


# -- now we get to the part specific for our problem

# how many independent sequences will we process in parallel?
BATCH_SIZE = 4

# what is the maximum context length for predictions?
BLOCK_SIZE = 8

# how many batches to look at when running `evaluate_loss`
# 300 comes from karpathy
NUM_BATCHES_TO_EVAL = 100

# how often to evaluate the model
EVALUATE_EVERY_N_STEPS = 10000

MAX_STEPS = 100000

N_EMBED = 32


# note: get's stuck around 2.4 no matter what
def main() -> None:

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
    )

    # Let's now split up the data into train and validation sets

    # first 90% will be train, rest val
    train_val_ratio = 0.9

    n = int(train_val_ratio * len(encoded_input_text))

    train_data: Int64[Tensor, "num_samples"] = encoded_input_text[:n]
    val_data: Int64[Tensor, "num_samples"] = encoded_input_text[n:]

    print(f"Splitting {len(encoded_input_text)} input tokens into")
    print(f" - train: {len(train_data)}")
    print(f" - val: {len(val_data)}")

    m = BigramEnhancedLanguageModel(
        vocab_size=len(vocab.unique_elements),
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED,
    )
    m.to(device)

    print("Output before optimizing:")
    print(generate_and_decode_text(m, vocab, device))

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # increase number of steps for good results...

    for step in range(MAX_STEPS):

        if step % EVALUATE_EVERY_N_STEPS == 0:
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

            print(
                f"Step {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

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

    print("Output after optimizing:")
    print(print(generate_and_decode_text(m, vocab, device)))


if __name__ == "__main__":
    main()
