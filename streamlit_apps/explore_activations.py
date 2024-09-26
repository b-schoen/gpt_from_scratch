import streamlit as st

import pandas as pd
import pathlib
import random
import string
import sys
import os
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import torch
import transformer_lens as tl

# This allows us to import modules from gpt_from_scratch even if it's not installed as a package

# Add the parent directory (which contains gpt_from_scratch) to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from gpt_from_scratch.serializable_activation_info import HookNameIndex, SampleIdentifier, HookName
from gpt_from_scratch import (
    saved_model_identifier,
    serializable_activation_info,
    naive_tokenizer,
)

Figure = Any


class SpecialToken:
    # note: as assume a BOS token because transformerlens expects it
    BOS = "<"
    # we use a EOS token for convenience
    EOS = ">"


def make_tokenizer() -> naive_tokenizer.NaiveTokenizer:
    """Make the same tokenizer used in toy_problem_hooked_transformer.ipynb"""

    vocab = string.ascii_lowercase + "|" + SpecialToken.BOS + SpecialToken.EOS

    return naive_tokenizer.NaiveTokenizer.from_text(vocab)


class Defaults:
    RUN_IDENTIFIER = "athletic_innocent_wren_of_intensity"
    INPUT_STRING = "<adcb|abcd"


def generate_all_partial_strings(input_string: str) -> list[str]:
    input_strings = [input_string]
    while input_string[-1] != "|":
        input_string = input_string[:-1]
        input_strings.append(input_string)

    # reverse, so they're in the order the model sees it
    input_strings.reverse()

    return input_strings


def add_resolved_columns_to_df(
    df: pd.DataFrame,
    activation_info: serializable_activation_info.SerializableActivationInfo,
) -> pd.DataFrame:
    """Add things that are expensive to compute or store, intended for use with small dataframes."""

    print(f"Adding resolved columns to df of len {len(df)}")

    hook_name_index_to_hook_name = {
        v: k for k, v in activation_info.hook_name_to_hook_name_index.items()
    }

    sample_identifier_to_token_string = {
        v: k for k, v in activation_info.token_string_to_sample_identifier.items()
    }

    # make a copy, since we're setting
    df = df.copy()

    df["hook_name"] = df["hook_name_index"].map(hook_name_index_to_hook_name)
    df["token_string"] = df.apply(
        lambda x: sample_identifier_to_token_string[(x["batch_index"], x["sample_in_batch_index"])],
        axis=1,
    )
    df["token"] = df.apply(lambda x: x["token_string"][int(x["position"])], axis=1)

    # filter out any activations that are 0
    #
    # note: these are always positive because ReLU
    df = df[df["activation_value"] > 0]

    print(f"Added resolved columns to df of len {len(df)}")

    return df


def generate_enriched_dataframe_for_input_string(
    input_string: str,
    activation_info: serializable_activation_info.SerializableActivationInfo,
) -> pd.DataFrame:

    input_strings = generate_all_partial_strings(input_string)

    dfs = []

    for input_string in tqdm.tqdm(input_strings):

        sample_identifier = activation_info.token_string_to_sample_identifier[input_string]

        # lookup activation data
        sample_df = activation_info.activation_df[
            (activation_info.activation_df["batch_index"] == sample_identifier[0])
            & (activation_info.activation_df["sample_in_batch_index"] == sample_identifier[1])
        ].copy()

        dfs.append(sample_df)

    df = pd.concat(dfs, ignore_index=True)

    df = add_resolved_columns_to_df(df, activation_info)

    return df


@st.cache_resource
def load_activation_info(
    run_identifier_string: str,
) -> serializable_activation_info.SerializableActivationInfo:

    with st.spinner("Loading activation info..."):

        identifier = saved_model_identifier.SavedModelsIdentifier.load_existing(
            run_identifier_string
        )

        activation_info = serializable_activation_info.SerializableActivationInfo.load(identifier)

    return activation_info


def load_token_string_to_sample_identifier(
    run_identifier_string: str,
) -> dict[str, SampleIdentifier]:

    activation_info = load_activation_info(run_identifier_string)

    return activation_info.token_string_to_sample_identifier


def load_hook_name_to_hook_name_index(
    run_identifier_string: str,
) -> dict[str, SampleIdentifier]:

    activation_info = load_activation_info(run_identifier_string)

    return activation_info.hook_name_to_hook_name_index


@st.cache_data
def load_enriched_dataframe(run_identifier_string: str, input_string: str) -> pd.DataFrame:

    activation_info = load_activation_info(run_identifier_string)

    # Display basic information
    st.write(f"Loaded activation info for run: {run_identifier_string}")
    st.write(f"Number of hooks: {len(activation_info.hook_name_to_hook_name_index)}")
    st.write(f"Number of samples: {len(activation_info.token_string_to_sample_identifier)}")

    # Display a sample of the activation dataframe
    st.subheader("Sample of Activation Data")
    st.dataframe(activation_info.activation_df.head())

    with st.spinner("Generating enriched dataframe..."):

        sample_df = generate_enriched_dataframe_for_input_string(
            input_string,
            activation_info,
        )

    return sample_df


def show_markdown_summary_of_dataframe_for_prompts(enriched_df: pd.DataFrame) -> None:
    """Show summary of data for asking models questions about visualization."""

    st.dataframe(enriched_df)

    markdown_str = """
The following is a summary of the activation data for the given input string.
This is for a small transformer model that is trained to predict the sorted order of a
small string. For example, if the input string is `<adcb|`, the model is trained to
predict `<adcb|abcd`. The activations recorded here are for the various hooks in the
model at every point in the residual stream for each token in the string. These
activations can be used to understand what the model is doing.

The transformer model is a `transformer_lens.HookedTransformer` with the following
architecture:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Hook Name                      ┃ Shape                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ hook_embed                     │ torch.Size([1, 10, 16])    │
│ hook_pos_embed                 │ torch.Size([1, 10, 16])    │
│ blocks.0.hook_resid_pre        │ torch.Size([1, 10, 16])    │
│ blocks.0.attn.hook_q           │ torch.Size([1, 10, 2, 8])  │
│ blocks.0.attn.hook_k           │ torch.Size([1, 10, 2, 8])  │
│ blocks.0.attn.hook_v           │ torch.Size([1, 10, 2, 8])  │
│ blocks.0.attn.hook_attn_scores │ torch.Size([1, 2, 10, 10]) │
│ blocks.0.attn.hook_pattern     │ torch.Size([1, 2, 10, 10]) │
│ blocks.0.attn.hook_z           │ torch.Size([1, 10, 2, 8])  │
│ blocks.0.hook_attn_out         │ torch.Size([1, 10, 16])    │
│ blocks.0.hook_resid_mid        │ torch.Size([1, 10, 16])    │
│ blocks.0.mlp.hook_pre          │ torch.Size([1, 10, 64])    │
│ blocks.0.mlp.hook_post         │ torch.Size([1, 10, 64])    │
│ blocks.0.hook_mlp_out          │ torch.Size([1, 10, 16])    │
│ blocks.0.hook_resid_post       │ torch.Size([1, 10, 16])    │
│ blocks.1.hook_resid_pre        │ torch.Size([1, 10, 16])    │
│ blocks.1.attn.hook_q           │ torch.Size([1, 10, 2, 8])  │
│ blocks.1.attn.hook_k           │ torch.Size([1, 10, 2, 8])  │
│ blocks.1.attn.hook_v           │ torch.Size([1, 10, 2, 8])  │
│ blocks.1.attn.hook_attn_scores │ torch.Size([1, 2, 10, 10]) │
│ blocks.1.attn.hook_pattern     │ torch.Size([1, 2, 10, 10]) │
│ blocks.1.attn.hook_z           │ torch.Size([1, 10, 2, 8])  │
│ blocks.1.hook_attn_out         │ torch.Size([1, 10, 16])    │
│ blocks.1.hook_resid_mid        │ torch.Size([1, 10, 16])    │
│ blocks.1.mlp.hook_pre          │ torch.Size([1, 10, 64])    │
│ blocks.1.mlp.hook_post         │ torch.Size([1, 10, 64])    │
│ blocks.1.hook_mlp_out          │ torch.Size([1, 10, 16])    │
│ blocks.1.hook_resid_post       │ torch.Size([1, 10, 16])    │
└────────────────────────────────┴────────────────────────────┘

Below is the collected set of activations for a given input string as it is progressively further
along in the completion process.

Our goal is to generate python visualization code to easily understand how these features and
activations are used to complete the task of sorting this string. We can get creative with
the exploration and visualizations, but the priority is on them being effective for understanding
and piecing together how the model works mechanistically using this data.

Our data has the following information:

| Column                | Description             |
|:----------------------|:------------------------|
| feature_index         | Index of the feature that is activating at this specific `hook_name` (ex: feature `17` at `hook_name` `blocks.1.hook_resid_post`)                       |
| activation_value      | Numeric value for the given feature, at the given hook_name, at the given position      |
| position              | Position in the token string sequence (ex: position `1` in `<a` is `a`) |
| k                     | Index in the `top_k` for the top (0 to 4) activations for the given feature, at the given hook_name, at the given position                      |
| hook_name_index       | Can be ignored                       |
| batch_index           | Can be ignored                       |
| sample_in_batch_index | Can be ignored                   |
| token_string          | The actual token string given as input to the model (ex: `<adcb|`) |
| hook_name             | The position in the `transformer_lens.HookedTransformer` model. This model is a small GPT-2 architecture model (decoder only transformer). (ex: `blocks.0.hook_resid_pre`) |

"""

    markdown_str += "\n\n[Example row]\n"

    markdown_str += enriched_df.iloc[0].to_markdown()

    markdown_str += "\n\n[Description]\n"

    markdown_str += enriched_df.describe().to_markdown()

    markdown_str += "\n\n[Info]\n"

    # why does this write to string lmao
    # Create a StringIO object to capture the DataFrame info
    buffer = io.StringIO()
    enriched_df.info(buf=buffer)
    # Get the captured output as a string and append it to markdown_str
    markdown_str += buffer.getvalue()

    for column in enriched_df.columns:

        # if not that many (arbitrary threshold) show all unique values
        if enriched_df[column].nunique() < 10:

            markdown_str += f"\n\n[Column: `{column}` unique values]\n"

            markdown_str += str(enriched_df[column].unique().tolist())

        else:

            markdown_str += f"\n\n[Column: `{column}` description]\n"

            markdown_str += str(enriched_df[column].describe())

    st.code(markdown_str, language="markdown")


# note: `title` is passed in for telling them apart in gifs etc
def plot_attentions_patterns(
    input_token_str_to_cache_dict: dict[str, tl.ActivationCache],
    title: str,
) -> Figure:
    """
    Visualize attention patterns for all layers and heads in the model for multiple caches.

    Args:
        caches (List[Dict[str, Any]]): List of caches containing attention patterns from model forward passes.

    Returns:
        plt.Figure: A matplotlib figure containing the visualized attention patterns.
    """
    input_token_strings = list(input_token_str_to_cache_dict.keys())
    caches = list(input_token_str_to_cache_dict.values())

    # Find all attention pattern tensors in the first cache (assuming all caches have the same structure)
    pattern_keys = [key for key in caches[0].keys() if key.endswith(".attn.hook_pattern")]

    n_layers = len(pattern_keys)
    n_heads = caches[0][pattern_keys[0]].shape[1]
    n_caches = len(caches)

    # Calculate total number of subplots
    total_subplots = n_layers * n_heads

    # Create a figure with subplots stacked vertically for each cache
    fig, axes = plt.subplots(n_caches, total_subplots, figsize=(4 * total_subplots, 4 * n_caches))

    # Set overall figure title
    fig.suptitle(title, fontsize=16)

    # Color maps for alternating heads
    cmaps = ["Blues", "Reds"]

    for cache_idx, cache in enumerate(caches):
        input_token_string = input_token_strings[cache_idx]
        for layer, key in enumerate(pattern_keys):
            attention_pattern = cache[key]

            # Remove batch dimension and move to CPU
            reshaped_pattern = attention_pattern.squeeze(0).detach().cpu().numpy()

            for head in range(n_heads):
                subplot_index = layer * n_heads + head
                ax = axes[cache_idx, subplot_index] if n_caches > 1 else axes[subplot_index]

                # Plot the attention pattern
                im = ax.imshow(reshaped_pattern[head], cmap=cmaps[head % len(cmaps)])

                # Set title for each subplot
                ax.set_title(f"L{layer}-H{head}", fontsize=8)

                # Set column labels as individual characters from input_token_string at the top
                ax.xaxis.tick_top()
                ax.set_xticks(range(len(input_token_string)))
                ax.set_xticklabels(list(input_token_string), fontsize=6, ha="right")

                ax.set_yticks([])  # Remove y-axis ticks

    plt.tight_layout()

    # close figure so doesn't keep taking up memory
    # plt.close(fig)

    return fig


def get_top_activating_examples_df(
    run_identifier_string: str,
    hook_name: str,
    feature_index: int,
    samples_per_quintile: int = 3,  # Number of samples to take from each quintile
    num_quintiles: int = 5,
    top_n: int = 5,
    bottom_n: int = 5,
    random_state: int = 42,  # For reproducibility
) -> pd.DataFrame:

    print(f"Loading activation info for {hook_name=}, {feature_index=}, {run_identifier_string=}")
    activation_info = load_activation_info(run_identifier_string)

    df = activation_info.activation_df

    hook_name_index = activation_info.hook_name_to_hook_name_index[hook_name]

    # lookup this specific selection
    print(f"Looking up {hook_name=}, {feature_index=}")
    df = df[(df["hook_name_index"] == hook_name_index) & (df["feature_index"] == feature_index)]

    df = df.sort_values("activation_value", ascending=False)

    # Select top_n and bottom_n
    top_n_df = df.head(top_n).copy()
    bottom_n_df = df.tail(bottom_n).copy()

    # Exclude top_n and bottom_n from the quintile sampling to avoid duplication
    df = df.drop(top_n_df.index).drop(bottom_n_df.index)

    # Create quintiles based on 'activation_value'
    print("Assigning quintiles based on 'activation_value'...")
    df["selection_type"] = pd.qcut(
        df["activation_value"], num_quintiles, labels=False, duplicates="drop"
    )

    # Sample from each quintile
    print(f"Sampling {samples_per_quintile} examples from each quintile...")
    df = (
        df.groupby("selection_type")
        .apply(
            lambda x: x.sample(
                n=min(samples_per_quintile, len(x)),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    # use `selection_type` to tell why selected
    df["selection_type"] = df["selection_type"].apply(lambda x: f"qcut: {x}")
    top_n_df["selection_type"] = "top_n"
    bottom_n_df["selection_type"] = "bottom_n"

    # Combine all sampled data
    df = pd.concat([top_n_df, bottom_n_df, df], ignore_index=True)

    # add things like hook name, token string, and token
    print("Adding resolved columns...")
    df = add_resolved_columns_to_df(df, activation_info)

    return df


def filter_df(df: pd.DataFrame, series: pd.Series, message: str) -> pd.DataFrame:

    len_df_before = len(df)

    df = df[series]

    st.write(f"{message}: {len_df_before} -> {len(df)} ({len(df) / len_df_before * 100:.2f}%)")

    return df


def make_selectbox_for_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:

    selected_column_value = st.sidebar.selectbox(f"Select {column}", df[column].unique())

    df = filter_df(
        df,
        df[column] == selected_column_value,
        f"{column}: {selected_column_value}",
    )

    return df, selected_column_value


@st.cache_data
def generate_attention_patterns_figure_for_input_string(
    input_string: str, run_identifier: str
) -> Any:

    device = tl.utils.get_device()

    identifier = saved_model_identifier.SavedModelsIdentifier.load_existing(
        run_identifier=run_identifier,
    )

    loaded_models = saved_model_identifier.load_run_models(identifier, device=device)

    model = loaded_models.model
    model.eval()

    tokenizer = make_tokenizer()

    with st.spinner("Getting full activation cache for input strings..."):

        tokens = torch.tensor(tokenizer.encode(input_string)).to(device)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda x: x.endswith(".attn.hook_pattern"),
        )

    return plot_attentions_patterns({input_string: cache}, title="Attention Patterns")


def main() -> None:

    # Set up the page configuration
    st.set_page_config(page_title="Activation Explorer", layout="wide")

    # Title of the app
    st.title("Activation Explorer")

    # Sidebar for user input
    st.sidebar.header("Configuration")

    # Input for run identifier
    run_identifier = st.text_input("Enter Run Identifier", Defaults.RUN_IDENTIFIER)

    if not run_identifier:
        st.info("Please enter a Run Identifier in the sidebar to start exploring.")
        return

    input_string = st.text_input("Enter input string", Defaults.INPUT_STRING)

    if not input_string:
        st.info("Please enter an input string in the sidebar to start exploring.")
        return

    # list some example strings the user could look at
    token_string_to_sample_identifier = load_token_string_to_sample_identifier(run_identifier)

    token_strings = list(token_string_to_sample_identifier.keys())

    example_token_strings = random.choices(token_strings, k=10)

    st.markdown("### Example token strings")

    st.code(example_token_strings)

    if st.checkbox("Show model activations cache:"):

        fig = generate_attention_patterns_figure_for_input_string(input_string, run_identifier)

        st.pyplot(fig)

    if st.checkbox("Show activations for input string:"):

        with st.spinner("Selecting activations..."):

            enriched_df = load_enriched_dataframe(
                run_identifier_string=run_identifier,
                input_string=input_string,
            )

        enriched_df = enriched_df.infer_objects()

        # fix types
        enriched_df["sample_in_batch_index"] = enriched_df["sample_in_batch_index"].astype(int)
        enriched_df["batch_index"] = enriched_df["batch_index"].astype(int)

        # note: we keep columns like hook_name_index around so we can join with larger
        #       dataframe if needed

        if st.checkbox("Show markdown summary of data"):
            show_markdown_summary_of_dataframe_for_prompts(enriched_df)

        enriched_df, selected_hook_name = make_selectbox_for_column(enriched_df, "hook_name")

        enriched_df, selected_token_string = make_selectbox_for_column(enriched_df, "token_string")

        # show top activations per position at this layer
        for position, df_position in enriched_df.groupby("position"):

            token = selected_token_string[position]

            st.write(f"{position} - `{token}`")

            st.write(df_position[["token", "feature_index", "activation_value", "k"]])

        enriched_df, selected_feature_index = make_selectbox_for_column(
            enriched_df, "feature_index"
        )

        st.write(enriched_df[["token", "position", "feature_index", "activation_value", "k"]])

        # now get the top activating examples for the selected feature
        with st.spinner("Getting top activating examples..."):
            top_activating_examples_df = get_top_activating_examples_df(
                run_identifier,
                selected_hook_name,
                selected_feature_index,
                top_n=1000,
            )

        st.markdown("### Top activating examples (~20 from each quintile)")

        st.code(top_activating_examples_df.describe().to_markdown())

        top_activating_examples_df["activation_value"] = top_activating_examples_df[
            "activation_value"
        ].astype(float)

        top_activating_examples_df = top_activating_examples_df.sort_values(
            "activation_value", ascending=False
        )

        # Iterate directly over the sorted DataFrame rows
        prev_selection_type = None

        for _, row in top_activating_examples_df.iterrows():

            token_string = row["token_string"]
            position = row["position"]
            activation_value = row["activation_value"]
            selection_type = row["selection_type"]

            # mark changes in selection type, as there's samples in between them
            if selection_type != prev_selection_type:
                st.markdown("...")
            prev_selection_type = selection_type

            markdown_str = f"[{selection_type}]\t({activation_value:.2f})\t"

            for token_index, token in enumerate(token_string):

                # Highlight the token at the specified position
                if token_index == position:
                    markdown_str += f":red[{token}] "
                else:
                    markdown_str += f"{token} "

            st.markdown(markdown_str)

        # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.info("This app allows you to explore activation data from saved model runs.")


if __name__ == "__main__":
    main()
