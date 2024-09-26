"""Utilities for saving and loading a set of models from disk"""

import dataclasses
import json
import torch

import coolname
import pathlib

from gpt_from_scratch import transcoder, sae


import transformer_lens as tl

import os


@dataclasses.dataclass
class SavedModelsIdentifier:
    """Make an identifier that creates it's own name, to make accidental overwriting hard"""

    run_identifier: str

    @property
    def run_dir(self) -> pathlib.Path:
        return pathlib.Path(os.path.expanduser("~/gpt_from_scratch/models")) / self.run_identifier

    @property
    def transcoders_dir(self) -> pathlib.Path:
        return self.run_dir / "transcoders"

    @property
    def saes_dir(self) -> pathlib.Path:
        return self.run_dir / "saes"

    @property
    def model_filepath(self) -> pathlib.Path:
        return self.run_dir / "model.pt"

    @property
    def metadata_filepath(self) -> pathlib.Path:
        return self.run_dir / "metadata.json"

    @classmethod
    def make_new(cls) -> "SavedModelsIdentifier":
        instance = cls(run_identifier=coolname.generate_slug().replace("-", "_"))

        instance.run_dir.mkdir(parents=True, exist_ok=False)
        instance.transcoders_dir.mkdir(parents=True, exist_ok=False)
        instance.saes_dir.mkdir(parents=True, exist_ok=False)

        return instance

    @classmethod
    def load_existing(cls, run_identifier: str) -> "SavedModelsIdentifier":
        instance = cls(run_identifier=run_identifier)

        if not instance.run_dir.exists():
            raise FileNotFoundError(f"Run directory {instance.run_dir} does not exist!")

        return instance


@dataclasses.dataclass
class LoadedModels:
    model: tl.HookedTransformer
    transcoder_per_hook: dict[str, transcoder.Transcoder]
    sae_per_hook: dict[str, sae.SAE]


def save_run_models(
    model: tl.HookedTransformer,
    transcoder_trainer_per_hook: dict[str, transcoder.TranscoderTrainer],
    sae_trainer_per_hook: dict[str, sae.SAETrainer],
    arbitrary_metadata_dict: dict[str, str] | None = None,
) -> SavedModelsIdentifier:

    identifier = SavedModelsIdentifier.make_new()

    print(f"Saving models to {identifier.run_dir}")

    # note: use the hook name so when loading can use filename as hook name

    for hook_name, trainer in transcoder_trainer_per_hook.items():
        filepath = identifier.transcoders_dir / f"{hook_name}.pt"
        print(f"Saving {filepath}")
        torch.save(trainer.transcoder, filepath)

    for hook_name, trainer in sae_trainer_per_hook.items():
        filepath = identifier.saes_dir / f"{hook_name}.pt"
        print(f"Saving {filepath}")
        torch.save(trainer.sae, filepath)

    print(f"Saving model: {identifier.model_filepath}...")
    torch.save(model, identifier.model_filepath)

    if arbitrary_metadata_dict:
        print(f"Saving arbitrary metadata: {identifier.metadata_filepath}...")
        with open(identifier.metadata_filepath, "w") as file:
            json.dump(arbitrary_metadata_dict, file)

    return identifier


def load_run_models(identifier: SavedModelsIdentifier, device: torch.device) -> LoadedModels:
    """
    Load models and trainers from the specified SavedModelsIdentifier.

    Note:
        We don't load `arbitrary_metadata_dict` because it shouldn't be used programatically

    Args:
        identifier (SavedModelsIdentifier): The identifier for the saved models.

    Returns:
        LoadedModels: An instance containing the loaded model, transcoders, SAEs, and metadata.
    """

    print(f"Loading models from {identifier.run_dir}")

    # Load the main model
    assert identifier.model_filepath.exists()

    print(f"Loading model from {identifier.model_filepath}...")
    model = torch.load(identifier.model_filepath, map_location=device, weights_only=False)

    # Load transcoders
    transcoder_per_hook = {}

    for filepath in identifier.transcoders_dir.glob("*.pt"):

        print(f"Loading {filepath}")
        transcoder_per_hook[filepath.stem] = torch.load(
            filepath, map_location=device, weights_only=False
        )

    # Load SAEs
    sae_per_hook = {}

    assert identifier.saes_dir.exists()

    for filepath in identifier.saes_dir.glob("*.pt"):
        print(f"Loading {filepath}")
        sae_per_hook[filepath.stem] = torch.load(filepath, map_location=device, weights_only=False)

    return LoadedModels(
        model=model,
        transcoder_per_hook=transcoder_per_hook,
        sae_per_hook=sae_per_hook,
    )
