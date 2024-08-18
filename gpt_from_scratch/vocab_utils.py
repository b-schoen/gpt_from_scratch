import dataclasses
from typing import Self, Iterable

# since python represents these this way
type Char = str
type EncodedChar = int


# TODO(bschoen): in general would want to abstract this based on type
class Vocabulary:

    def __init__(self, value: Iterable[str]) -> None:

        self.unique_elements: list[Char] = sorted(list(set(value)))

        # create encoding / decoding mapping
        self._char_to_int = {char: i for i, char in enumerate(self.unique_elements)}
        self._int_to_char = {i: char for i, char in enumerate(self.unique_elements)}

    def encode(self, string: str) -> list[EncodedChar]:
        return [self._char_to_int[c] for c in string]

    def decode(self, encoded_chars: list[EncodedChar]) -> str:
        return "".join([self._int_to_char[ec] for ec in encoded_chars])

    def decode_single(self, encoded_char: EncodedChar) -> Char:
        return self._int_to_char[encoded_char]

    def encode_single(self, char: Char) -> EncodedChar:
        return self._char_to_int[char]
