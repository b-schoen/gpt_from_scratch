from gpt_from_scratch.python_utils import (
    wraps_partial,
    convert_return_value_to_json_string_wrapper,
)
from gpt_from_scratch.tokenizer_utils import (
    get_detailed_and_complete_tokenization_info_for_text,
    TokenInfo,
)

import tiktoken

import json
import dataclasses


@dataclasses.dataclass(frozen=True)
class Info:
    name: str
    age: int


def generate_info(name: str, age: int) -> Info:
    return Info(name=name, age=age)


def foo(bar: int, buzz: str = "cat") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: cat).

    """

    return f"foo-{bar}-{buzz}"


def test_wraps_partial() -> None:

    wrapped_fn = wraps_partial(foo, 23)

    # check calling works as expected
    assert wrapped_fn(buzz="dog") == "foo-23-dog"

    # check docstring still there
    assert wrapped_fn.__doc__ == foo.__doc__

    # TODO(bschoen): Check name?


def test_convert_return_value_to_json_string_wrapper_for_string_type() -> None:
    """Test that string return type is passed through."""

    wrapped_fn = convert_return_value_to_json_string_wrapper(foo)

    assert wrapped_fn(bar=23, buzz="dog") == "foo-23-dog"


def test_convert_return_value_to_json_string_wrapper_for_non_string_type() -> None:
    """Test that non-string types are serialized"""

    wrapped_fn = convert_return_value_to_json_string_wrapper(generate_info)

    info = Info(name="John", age=42)

    result_json = wrapped_fn(name=info.name, age=info.age)

    result_dict = json.loads(result_json)

    deserialized_info = Info(**result_dict)

    assert info == deserialized_info


def test_combined_wraps_partial_and_convert_return_value_to_json_string_wrapper() -> (
    None
):
    # note: this is `gpt-4o`
    gpt4_tokenizer = tiktoken.get_encoding("o200k_base")

    text = "你好 strawberry"

    # pass tokenizer as partial
    wrapped_with_tokenizer_fn = wraps_partial(
        get_detailed_and_complete_tokenization_info_for_text,
        gpt4_tokenizer,
    )

    # convert return type
    wrapped_with_return_type_serialization_fn = (
        convert_return_value_to_json_string_wrapper(wrapped_with_tokenizer_fn)
    )

    # call without wrappers
    token_infos = get_detailed_and_complete_tokenization_info_for_text(
        tokenizer=gpt4_tokenizer,
        text=text,
    )

    # call with wrappers
    response_from_wrapped_fn = wrapped_with_return_type_serialization_fn(text=text)

    # deserialize JSON
    response_dict = json.loads(response_from_wrapped_fn)

    # deserialize dataclasses
    deserialized_token_infos = [TokenInfo(**x) for x in response_dict]

    # check that we get the same result
    assert token_infos == deserialized_token_infos
