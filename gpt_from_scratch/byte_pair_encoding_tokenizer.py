import dataclasses
from typing import Self, Sequence
import collections
import pathlib
import regex as re

type TokenInt = int
type Byte = bytes

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
def merge[T](tokens: list[T], pair: tuple[T, T], new_token: T) -> list[T]:
    """
    Merge a pair of tokens into a new token.

    Note:
        - For character level models, `T` is `int`
        - For anything above character level models, `T` is `bytes` (a word)

    """
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
    """Byte Pair Encoding Tokenizer using BPE merge (over characters)."""

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

        tokens: list[int] = list(input_bytes)

        # first unused int value (since bytes are 0..255)
        first_unused_int_value: int = 256

        # num merges is just vocab size - assumption that we start with each byte
        # encoded
        num_merges = vocab_size - first_unused_int_value

        original_token_length = len(tokens)

        print(f"Constructing {num_merges} merges from {len(tokens)} unmerged tokens...")
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

    # for compatibility with tiktoken
    def decode_single_token_bytes(self, encoded_byte: int) -> bytes:
        return self.vocab[encoded_byte]


def bpe_encode(
    mergeable_ranks: dict[bytes, int],
    input_bytes: bytes,
) -> list[TokenInt]:
    parts = [bytes([b]) for b in input_bytes]
    while True:
        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break

        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )

    tokens = [mergeable_ranks[part] for part in parts]

    return tokens


class RegexSplitPatterns:

    # TODO(bschoen): More general handling of special tokens? Is this okay? We actually
    #                still want to split this (so it needs to be in the regex pattern)
    #                but we don't want them merged during `bpe_merge`
    #
    #                This means we can assume that special tokens come in already split
    #                as an exact match (even if in a public api we'd want to construct
    #                this regex automatically for the user given special tokens)
    CUSTOM_TINYSTORIES = "|".join(
        [
            # Match whole words
            #
            #   \b    - Represents a word boundary (transition from a non-word char to
            #            a word char or vice versa)
            #   \w+   - Matches one or more word characters (letters, digits, or underscores)
            #   \b    - Another word boundary to ensure we match whole words
            #
            r"\b\w+\b",
            #
            # Match single punctuation marks
            #
            #   []       - Character set: match any single character listed inside the brackets
            #   .,!?;:"  - The actual characters we want to match (various punctuation marks)
            #
            r'[.,!?;:"]',
            #
            # Match one or more whitespace characters  (spaces, tabs)
            #
            r"\s+",
            #
            # Match the newline character
            #
            r"\n",
            #
            # Match the special end-of-text token exactly
            #
            r"<\|endoftext\|>",
        ]
    )


@dataclasses.dataclass(frozen=True)
class BytePairEncodingWordTokenizer:
    """
    Basically the same as `BytePairEncodingTokenizer`, but we're doing BPE on each word separately.

    Note:
        - This class gets *extremely* unnecessarily confusing due to python
          not having a clear distinction between `bytes` and `list[<single byte>]`
        - This whole class is a nightmare because `_educational.py` uses a mix
          of `bytes` and `list[bytes]` to represent ints, whereas Karpathy
          used `int` for everything.
        - `_educational.py` also uses `min_rank` to keep track of when merges
          are done, instead of just not having them in the `merges` dict

    """

    # TODO(bschoen): This is confusing, since previous class used mixed K/V,
    #                but more consistent with reference implementation using `rank`

    # NOTE: These are just inverted dicts with the same information, which
    #       is useful for quickly encoding/decoding

    # ex: {278: b'th', 279: b'in', ...}
    vocab: dict[TokenInt, bytes]

    # note: importantly, allows increasing string size, ex: `i` + `ng`
    #
    # ex: {(b't', b'h'): 278, (b'i', b'n'): 279, ...}
    merges: dict[bytes, TokenInt]

    # note: we leave this as a string for easy serialization / deserialization
    #       of the tokenizer, with the slight performance cost of having to
    #       recompile during encode, but that's fairly infrequent in training
    #       and worth avoiding the cost of complexity
    regex_split_pattern_string: str

    special_tokens_dict: dict[str, TokenInt]

    # TODO(bschoen): Name this `train`? That gets really confusing, still
    #                want it to be general though (as a larger rule,
    #                like if substituting types.
    @classmethod
    def from_input_text(
        cls,
        input_text: str,
        regex_split_pattern_string: str,
        vocab_size: int,
        special_tokens: frozenset[str] | None = None,
    ) -> Self:
        """
        Useful for anything above character level models.

        Note:
            - We still start with individual bytes
            - Roughly based on a combination of Karpathy's lecture and
              https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py#L117

        """

        special_tokens = special_tokens or set()

        regex_split_pattern = re.compile(regex_split_pattern_string)

        # original bytes
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # split each word into list of bytes so we can apply bpe_merge
        #
        # split to where there's one byte per letter in each "word"
        words: list[list[Byte]] = [
            # skip special tokens(as we don't want those merged), and they're handled
            # already before the merges start when `special_tokens_dict` is constructed
            [bytes([b]) for b in word.encode("utf-8") if word not in special_tokens]
            for word in re.findall(regex_split_pattern, input_text)
        ]

        original_avg_word_length = sum(len(word) for word in words) / len(words)

        print(f"Constructing {vocab_size=} from {len(words)} unmerged words...")
        merges: dict[bytes, TokenInt] = {v: k for k, v in vocab.items()}

        # First add special tokens to vocab and merges
        special_tokens_dict: dict[str, TokenInt] = {}
        for token in special_tokens:
            if len(vocab) >= vocab_size:
                break
            new_token = len(vocab)
            token_bytes = token.encode("utf-8")
            vocab[new_token] = token_bytes
            merges[token_bytes] = new_token
            special_tokens_dict[token] = new_token

        # now iteratively apply merges of most frequent pair
        while len(vocab) < vocab_size:

            # Find the most common pair. This will become our next token
            #
            # note: This is also done slightly differently than `from_input_bytes`
            #       since we're doing it on a word level
            #
            # TODO(bschoen): We could modify previous counter by the result of merge
            #                instead of recomputing from scratch every time
            pair_counts: collections.Counter[tuple[Byte, Byte]] = collections.Counter()

            # ex: word = [b'h', b'e', b'l', b'l', b'o']
            for word in words:

                # ex: pair = (b'h', b'e')
                for pair in zip(word[:-1], word[1:]):

                    pair_counts[pair] += 1

            # ex: top_pair = (b't', b'h')
            top_pair = max(pair_counts, key=pair_counts.get)

            # ex: token_bytes = b'th' (adding concats them)
            merged_bytes: bytes = top_pair[0] + top_pair[1]

            # ex: new_token = 278
            new_token: TokenInt = len(vocab) + 1

            # keep track of merges (and their order)
            #
            # ex: merges[(b't', b'h')] = 278
            merges[merged_bytes] = new_token

            # same information, just kept conveniently for decoding
            vocab[new_token] = merged_bytes

            print(
                f"[vocab: {len(vocab)} / {vocab_size}] "
                f"Merging\t{top_pair}"
                f"\t(count: {pair_counts[top_pair]})"
                f"\t-> new token: {new_token}"
            )

            # only difference from `from_input_bytes` is that we
            # need to call `bpe_merge` for each word, since `bpe_merge` can't
            # cross word boundaries

            # now we know the most frequent, so apply it *within* each word
            new_words: list[list[Byte]] = []

            # ex: word = [b'h', b'e', b'l', b'l', b'o']
            for word in words:

                # THIS is where we get to re-use `merge`
                new_word = []
                i = 0
                while i < len(word) - 1:

                    if (word[i], word[i + 1]) == top_pair:

                        # We found our pair! Merge it
                        # print(f"Merging {word[i]} + {word[i + 1]} -> {merged_bytes}")
                        new_word.append(merged_bytes)
                        i += 2

                    else:
                        new_word.append(word[i])
                        i += 1

                if i == len(word) - 1:
                    new_word.append(word[i])

                new_words.append(new_word)

            words = new_words

        final_avg_word_length = sum(len(word) for word in words) / len(words)

        print(
            f"Length {original_avg_word_length} -> {final_avg_word_length} "
            f"- compression ratio: {original_avg_word_length / final_avg_word_length:.2f}X"
        )

        print(f"Constructed `{cls.__name__}`")
        return cls(
            vocab=vocab,
            merges=merges,
            regex_split_pattern_string=regex_split_pattern_string,
            special_tokens_dict=special_tokens_dict,
        )

    def encode(self, text: str) -> list[TokenInt]:

        # Use the regex to split the text into (approximately) words
        regex_split_pattern = re.compile(self.regex_split_pattern_string)
        words = regex_split_pattern.findall(text)

        tokens = []

        for word in words:

            # handle special tokens
            if word in self.special_tokens_dict:
                token = self.special_tokens_dict[word]
                tokens.append(token)
                continue

            # otherwise we've got non special tokens

            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")

            word_tokens = bpe_encode(
                mergeable_ranks=self.merges,
                input_bytes=word_bytes,
            )

            tokens.extend(word_tokens)

        return tokens

    # note: decode functions are both the same

    def decode(self, encoded_bytes: list[TokenInt]) -> str:

        # given ids (list of integers), return Python string
        vocab_byte_string = b"".join(
            self.vocab[encoded_byte] for encoded_byte in encoded_bytes
        )

        # replace with special marker (�) for any bytes that can't be decoded
        # UTF-8 requires special start tokens for multi-byte
        # standard practice is to use `errors="replace"`
        text = vocab_byte_string.decode("utf-8", errors="replace")

        return text

    # for compatibility with tiktoken
    def decode_single_token_bytes(self, encoded_byte: TokenInt) -> bytes:

        return self.vocab[encoded_byte]
