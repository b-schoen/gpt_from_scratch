import unicodedata
import random
from typing import Protocol

import termcolor
from colored import fg, bg, attr


class Tokenizer(Protocol):
    """Tokenizer protocol compatible with both `BytePairEncodingTokenizer` and `tiktoken.Encoding`"""

    def decode(self, encoded_bytes: list[int]) -> str: ...

    def encode(self, text: str) -> list[int]: ...

    def decode_single_token_bytes(self, encoded_byte: int) -> bytes: ...


def get_colored_tokenization_of_split_string(
    substrings: list[str],
) -> str:
    """Assuming we have a string split based on some tokenization, return coloring of it."""

    # Use a limited set of widely supported background colors
    bg_colors = ["on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan"]
    random.shuffle(bg_colors)

    colored_text = ""

    for i, substring in enumerate(substrings):
        token_string = substring

        # Color this token's background
        bg_color = bg_colors[
            i % len(bg_colors)
        ]  # Cycle through colors if we have more tokens than colors

        # Use black text on light backgrounds, white text on dark backgrounds
        # text_color = 'black' if bg_color in ['on_yellow', 'on_cyan'] else 'white'
        text_color = "white"

        colored_text += termcolor.colored(token_string, text_color, bg_color)

    return colored_text


def get_colored_tokenization(
    tokenizer: Tokenizer,
    input_string: str,
) -> str:
    """Get string with terminal colors marking the tokenization of the input string."""

    # Encode the input string
    encoded_tokens = tokenizer.encode(input_string)

    # Use a limited set of widely supported background colors
    bg_colors = ["on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan"]
    random.shuffle(bg_colors)

    colored_text = ""

    for i, token in enumerate(encoded_tokens):
        token_bytes = tokenizer.decode_single_token_bytes(token)
        token_string = token_bytes.decode("utf-8", errors="replace")

        # Color this token's background
        bg_color = bg_colors[
            i % len(bg_colors)
        ]  # Cycle through colors if we have more tokens than colors

        # Use black text on light backgrounds, white text on dark backgrounds
        # text_color = 'black' if bg_color in ['on_yellow', 'on_cyan'] else 'white'
        text_color = "white"

        colored_text += termcolor.colored(token_string, text_color, bg_color)

    return colored_text


def color_encode(byte: int) -> str:
    if byte < 128:  # ASCII
        return fg("green") + format(byte, "02X") + attr("reset")
    elif byte < 192:  # Continuation byte
        return fg("yellow") + format(byte, "02X") + attr("reset")
    elif byte < 224:  # Start of 2-byte sequence
        return fg("blue") + format(byte, "02X") + attr("reset")
    elif byte < 240:  # Start of 3-byte sequence
        return fg("magenta") + format(byte, "02X") + attr("reset")
    else:  # Start of 4-byte sequence
        return fg("red") + format(byte, "02X") + attr("reset")


def show_token_mapping(tokenizer: Tokenizer, input_string: str) -> None:
    """Display tokenization for the given input string."""

    # Color legend
    # print("Color legend:")
    # print(f"\t{fg('green')}Green{attr('reset')}:\t\tASCII")
    # print(f"\t{fg('yellow')}Yellow{attr('reset')}:\t\tUTF-8 continuation byte")
    # print(f"\t{fg('blue')}Blue{attr('reset')}:\t\tStart of 2-byte UTF-8 sequence")
    # print(f"\t{fg('magenta')}Magenta{attr('reset')}:\tStart of 3-byte UTF-8 sequence")
    # print(f"\t{fg('red')}Red{attr('reset')}:\t\tStart of 4-byte UTF-8 sequence")

    encoded_tokens = tokenizer.encode(input_string)

    print(f"Input:\t\t{input_string}")
    print(f"Tokenized:\t{get_colored_tokenization(tokenizer, input_string)}")

    print("Token ID | Token Bytes | Token String")
    print("---------+-------------+--------------")

    current_byte_position = 0
    current_char_position = 0

    for token in encoded_tokens:
        token_bytes = tokenizer.decode_single_token_bytes(token)
        token_string = token_bytes.decode("utf-8", errors="replace")

        # Color-code the bytes
        colored_bytes = " ".join(color_encode(b) for b in token_bytes)

        print(f"{token:8d} | {colored_bytes:11s} | '{token_string}'")

        # Find the corresponding characters in the original input string
        char_end_position = current_char_position
        while char_end_position < len(input_string) and current_byte_position + len(
            token_bytes
        ) > len(input_string[:char_end_position].encode("utf-8")):
            char_end_position += 1

        # Highlight the part of the input string that corresponds to this token
        highlighted_input = (
            input_string[:current_char_position]
            + bg("red")
            + fg("white")
            + input_string[current_char_position:char_end_position]
            + attr("reset")
            + input_string[char_end_position:]
        )
        print(f"          {highlighted_input}")

        # Print Unicode details for each character in the token
        for char in input_string[current_char_position:char_end_position]:
            char_name = unicodedata.name(char, "Unknown")
            char_bytes = char.encode("utf-8")
            colored_char_bytes = " ".join(color_encode(b) for b in char_bytes)
            print(
                f"          U+{ord(char):04X} {char_name} ({len(char_bytes)} bytes: {colored_char_bytes})"
            )

        # print()

        current_byte_position += len(token_bytes)
        current_char_position = char_end_position
