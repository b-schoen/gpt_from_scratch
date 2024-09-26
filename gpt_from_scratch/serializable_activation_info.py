import dataclasses
import pandas as pd
import json

import pathlib

from gpt_from_scratch import saved_model_identifier

BatchIndex = int
SampleInBatchIndex = int
HookName = str
HookNameIndex = int
SampleIdentifier = tuple[BatchIndex, SampleInBatchIndex]


@dataclasses.dataclass
class SerializableActivationInfo:
    """
    Serializable activation info used in toy_problem_hooked_transformer.ipynb

    This class allows us to easily save huge activation data from
    toy_problem_hooked_transformer.ipynb, and easily load it in a streamlit app for exploration.

    """

    identifier: saved_model_identifier.SavedModelsIdentifier

    hook_name_to_hook_name_index: dict[HookName, HookNameIndex]
    token_string_to_sample_identifier: dict[str, SampleIdentifier]
    activation_df: pd.DataFrame

    @property
    def directory(self) -> pathlib.Path:
        return self.identifier.run_dir / "activation_info"

    @property
    def hook_name_to_hook_name_index_filepath(self) -> pathlib.Path:
        return self.directory / "hook_name_to_hook_name_index.json"

    @property
    def token_string_to_sample_identifier_filepath(self) -> pathlib.Path:
        return self.directory / "token_string_to_sample_identifier.json"

    @property
    def activation_df_filepath(self) -> pathlib.Path:
        return self.directory / "activation_df.pkl.gz"

    @classmethod
    def load(
        cls,
        identifier: saved_model_identifier.SavedModelsIdentifier,
    ) -> "SerializableActivationInfo":

        # Create an instance of SerializableActivationInfo
        instance = cls(
            identifier=identifier,
            hook_name_to_hook_name_index={},
            token_string_to_sample_identifier={},
            activation_df=pd.DataFrame(),
        )

        print(f"Loading activation info from {instance.directory}")

        # Load hook_name_to_hook_name_index
        print(
            "Loading hook_name_to_hook_name_index from "
            f"{instance.hook_name_to_hook_name_index_filepath}"
        )
        with open(instance.hook_name_to_hook_name_index_filepath, "r") as f:
            instance.hook_name_to_hook_name_index = json.load(f)

        # Load token_string_to_sample_identifier
        print(
            "Loading token_string_to_sample_identifier from "
            f"{instance.token_string_to_sample_identifier_filepath}"
        )
        with open(instance.token_string_to_sample_identifier_filepath, "r") as f:
            serialized_data = json.load(f)
            # Convert lists back to tuples
            instance.token_string_to_sample_identifier = {
                k: tuple(v) for k, v in serialized_data.items()
            }

        # Load activation_df
        print(f"Loading activation_df from {instance.activation_df_filepath}")
        instance.activation_df = pd.read_pickle(instance.activation_df_filepath)

        print("Activation info loaded successfully")
        return instance

    def save(self) -> None:

        print(f"Saving activation info to {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)

        print(
            "Saving hook_name_to_hook_name_index to "
            f"{self.hook_name_to_hook_name_index_filepath}"
        )
        with open(self.hook_name_to_hook_name_index_filepath, "w") as f:
            json.dump(self.hook_name_to_hook_name_index, f)

        print(
            "Saving token_string_to_sample_identifier to "
            f"{self.token_string_to_sample_identifier_filepath}"
        )
        # Convert tuples to lists for JSON serialization
        serializable_token_string_to_sample_identifier = {
            k: list(v) for k, v in self.token_string_to_sample_identifier.items()
        }

        with open(self.token_string_to_sample_identifier_filepath, "w") as f:
            json.dump(serializable_token_string_to_sample_identifier, f)

        # Save activation_df

        # Pandas handles .pkl.gz automatically
        print(f"Saving activation_df to {self.activation_df_filepath}")
        self.activation_df.to_pickle(self.activation_df_filepath)

        print("Activation info saved successfully")
