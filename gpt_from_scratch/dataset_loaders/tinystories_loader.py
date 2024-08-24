import enum
import dataclasses
from typing import Iterator, Callable
import io
import pathlib

# let's load tinystories for comparison
#
# note: `datasets` can list datasets but is deprecated
import huggingface_hub

import torch
from torch.utils.data import IterableDataset, DataLoader


@dataclasses.dataclass(frozen=True)
class TrainAndVal[T]:
    """Helper for common pattern of transforming both train and val."""

    train: T
    val: T

    def apply[R](self, func: Callable[[T], R]) -> "TrainAndVal[R]":

        return dataclasses.replace(
            self,
            train=func(self.train),
            val=func(self.val),
        )


class TinyStoriesVersion(enum.Enum):
    # original in paper
    V1 = "TinyStories-"

    # GPT-4 only, significantly larger but newer
    V2 = "TinyStoriesV2-GPT4-"


def _get_tinystories_filenames(version: TinyStoriesVersion) -> TrainAndVal[str]:

    return TrainAndVal(
        train=f"{version.value}train.txt",
        val=f"{version.value}valid.txt",
    )


def _download_file_from_tinystories(filename: str) -> pathlib.Path:

    # from https://huggingface.co/docs/huggingface_hub/en/guides/download#from-latest-version
    print(f"Downloading {filename}...")
    filepath = huggingface_hub.hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename=filename,
        repo_type="dataset",
    )

    print(f"Downloaded {filename} to {filepath}")
    return pathlib.Path(filepath)


class TinyStoriesIterableDataset(IterableDataset):
    def __init__(self, filepath: pathlib.Path, tokenizer, chunk_size: int = 1000000):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                tokens = self.tokenizer.encode(chunk)
                yield from tokens


class TinyStoriesDataLoader:
    def __init__(
        self,
        B: int,
        T: int,
        filepath: pathlib.Path,
        tokenizer,
    ) -> None:
        self.B = B
        self.T = T
        self.dataset = TinyStoriesIterableDataset(filepath, tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=B * T + 1, drop_last=True)
        self.iterator = iter(self.dataloader)

        # Estimate total tokens
        with open(filepath, "r", encoding="utf-8") as f:
            file_size = f.seek(0, io.SEEK_END)

        estimated_tokens = (
            file_size // 4
        )  # Rough estimate, assuming 4 bytes per token on average

        print(f"Estimated {estimated_tokens} tokens")
        print(
            f"Estimated 1 epoch = {estimated_tokens // (B * T)} "
            "batches (steps to make one pass through data)"
        )

    def next_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)

        return x, y


def download_tinystories(version: TinyStoriesVersion) -> TrainAndVal[pathlib.Path]:

    filenames = _get_tinystories_filenames(version)
    filepaths = filenames.apply(_download_file_from_tinystories)

    return filepaths


def load_tinystories(
    version: TinyStoriesVersion,
    B: int,
    T: int,
    tokenizer,
) -> TrainAndVal[TinyStoriesDataLoader]:
    filepaths = download_tinystories(version)

    return TrainAndVal(
        train=TinyStoriesDataLoader(B, T, filepaths.train, tokenizer),
        val=TinyStoriesDataLoader(B, T, filepaths.val, tokenizer),
    )
