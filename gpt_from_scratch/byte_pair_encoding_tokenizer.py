import dataclasses
from typing import Self, Sequence
import collections
import pathlib


# so the strategy is we'll start merging
# - we know we can start at 256, since
#   the bytes only go up to 255


def pair_to_char(pair: tuple[int, int]) -> tuple[str, str]:
    return (chr(pair[0]), chr(pair[1]))


def get_pair_counts[T](sequence: Sequence[T]) -> dict[tuple[T, T], int]:

    # default count to 0
    pair_counts: dict[tuple[T, T], int] = collections.defaultdict(int)

    for current_element, next_element in zip(sequence, sequence[1:]):

        pair_counts[(current_element, next_element)] += 1

    return pair_counts


# TODO(bschoen): This has got to be insanely slow
#                - probably fine since only doing once
def merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
    # print(f'Merging {pair} [{pair_to_char(pair)}] -> {replacement_token}')

    # in the list of ints (tokens), replace all consecutive occurences
    # of pair with the new token idx
    tokens_after_merge = []
    i = 0

    while i < len(tokens):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(tokens) - 1 and pair == (tokens[i], tokens[i + 1]):
            tokens_after_merge.append(new_token)
            i += 2
        else:
            tokens_after_merge.append(tokens[i])
            i += 1
    return tokens_after_merge


def show_top_pair_counts(
    pair_counts: dict[tuple[int, int], int],
    top_n: int,
) -> None:
    pair_counts_sorted = {
        k: v for k, v in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    }

    print("<pair>\t\t<count>\t<decoded>")
    print("---\t\t---\t---")
    for pair, count in list(pair_counts_sorted.items())[:top_n]:
        print(f"{pair}\t{count}\t{pair_to_char(pair)}")


@dataclasses.dataclass(frozen=True)
class BytePairEncodingTokenizer:

    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]

    @classmethod
    def from_input_bytes(
        cls,
        # ex: `text.encode("utf-8")`
        input_bytes: bytes,
        # note: vocabulary size is a hyperparameter
        # ex: GPT-4 uses ~100,000 tokens
        vocab_size: int,
    ) -> Self:

        # first unused int value (since bytes are 0..255)
        first_unused_int_value: int = 256

        num_merges = vocab_size - first_unused_int_value

        tokens: list[int] = list(input_bytes)
        original_token_length = len(tokens)

        print(f"Constructing {num_merges} merges...")
        merges: dict[tuple[int, int], int] = {}

        for i in range(num_merges):

            # TODO(bschoen): could optimize since only actually want top pair
            # TODO(bschoen): the only other place this is used (`encode`) it's using `min`
            pair_counts = get_pair_counts(tokens)

            top_pair = max(pair_counts, key=pair_counts.get)

            new_token = first_unused_int_value + i

            print(
                f"Merging\t{top_pair}"
                f"\t[{pair_to_char(top_pair)}]"
                f"\t(count: {pair_counts[top_pair]})"
                f"\t-> new token: {new_token}"
            )

            tokens = merge(tokens=tokens, pair=top_pair, new_token=new_token)

            # keep track of merges (and their order)
            merges[top_pair] = new_token

        print(
            f"Length {original_token_length} "
            f"-> {len(tokens)} "
            f"- compression ratio: {original_token_length / len(tokens):.2f}X"
        )

        print("Constructing vocab...")

        # original bytes
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # new tokens created during merges
        # note: really matters that this runs in order that the replacements happened
        for (p0, p1), new_token in merges.items():
            vocab[new_token] = vocab[p0] + vocab[p1]

        print("Constructed `BytePairEncodingTokenizer`")
        return cls(vocab=vocab, merges=merges)

    def decode(self, encoded_bytes: list[int]) -> str:

        # given ids (list of integers), return Python string
        vocab_byte_string = b"".join(
            self.vocab[encoded_byte] for encoded_byte in encoded_bytes
        )

        # replace with special marker (�) for any bytes that can't be decoded
        # UTF-8 requires special start tokens for multi-byte
        # standard practice is to use `errors="replace"`
        text = vocab_byte_string.decode("utf-8", errors="replace")

        return text

    def encode(self, text: str) -> list[int]:

        # given a string, return list of integers (the tokens)
        tokens: list[int] = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pair_counts = get_pair_counts(tokens)

            # essentially going in reverse order?
            top_pair = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))

            if top_pair not in self.merges:
                # nothing else can be merged
                break

            new_token = self.merges[top_pair]

            tokens = merge(tokens, top_pair, new_token)

        return tokens
