from gpt_from_scratch import (
    tokenizer_utils,
    naive_tokenizer,
)

from gpt_from_scratch.tokenizer_utils import TokenInfo

import string


def test_naive_tokenizer_on_ascii() -> None:

    vocab = string.ascii_letters.lower() + "|"

    tokenizer = naive_tokenizer.NaiveTokenizer.from_text(vocab)

    input_text = "hello|olleh"

    tokenization_info = (
        tokenizer_utils.get_detailed_and_complete_tokenization_info_for_text(
            tokenizer, input_text
        )
    )

    assert tokenization_info == [
        TokenInfo(token_index=0, token_string="h", token_split_into_characters=["h"]),
        TokenInfo(token_index=1, token_string="e", token_split_into_characters=["e"]),
        TokenInfo(token_index=2, token_string="l", token_split_into_characters=["l"]),
        TokenInfo(token_index=3, token_string="l", token_split_into_characters=["l"]),
        TokenInfo(token_index=4, token_string="o", token_split_into_characters=["o"]),
        TokenInfo(token_index=5, token_string="|", token_split_into_characters=["|"]),
        TokenInfo(token_index=6, token_string="o", token_split_into_characters=["o"]),
        TokenInfo(token_index=7, token_string="l", token_split_into_characters=["l"]),
        TokenInfo(token_index=8, token_string="l", token_split_into_characters=["l"]),
        TokenInfo(token_index=9, token_string="e", token_split_into_characters=["e"]),
        TokenInfo(token_index=10, token_string="h", token_split_into_characters=["h"]),
    ]
