from gpt_from_scratch.tokenizer_utils import (
    get_detailed_and_complete_tokenization_info_for_text,
    TokenInfo,
)

import tiktoken


def test_get_detailed_and_complete_tokenization_info_for_text() -> None:

    # note: this is `gpt-4o`
    gpt4_tokenizer = tiktoken.get_encoding("o200k_base")

    text = "你好 strawberry"

    token_infos = get_detailed_and_complete_tokenization_info_for_text(
        tokenizer=gpt4_tokenizer,
        text=text,
    )

    expected_token_infos = [
        TokenInfo(
            token_index=0,
            token_string="你好",
            token_split_into_characters=["你", "好"],
        ),
        TokenInfo(
            token_index=1,
            token_string=" strawberry",
            token_split_into_characters=[
                " ",
                "s",
                "t",
                "r",
                "a",
                "w",
                "b",
                "e",
                "r",
                "r",
                "y",
            ],
        ),
    ]

    assert token_infos == expected_token_infos
