import dataclasses


TokenInt = int
Byte = bytes


@dataclasses.dataclass(frozen=True)
class NaiveTokenizer:
    """
    Naive tokenizer that maps each unique byte to a unique token

    Satisfies the tiktoken interface (and tiktoken_utils.Tokenizer protocol)

    Useful for toy problems

    """

    # note: store both for fast lookup during both encoding and decoding
    byte_to_token_dict: dict[Byte, TokenInt]
    token_to_byte_dict: dict[TokenInt, Byte]

    @classmethod
    def from_text(cls, text: str) -> "NaiveTokenizer":

        byte_to_token_dict: dict[Byte, TokenInt] = {}

        text_as_bytes = text.encode("utf-8")

        unique_bytes = set(text_as_bytes)

        for index, unique_byte in enumerate(unique_bytes):
            byte_to_token_dict[bytes([unique_byte])] = index

        # create inverse lookup for decoding
        token_to_byte_dict = {v: k for k, v in byte_to_token_dict.items()}

        return cls(
            byte_to_token_dict=byte_to_token_dict,
            token_to_byte_dict=token_to_byte_dict,
        )

    def encode(self, text: str) -> list[int]:

        text_as_bytes = text.encode("utf-8")

        return [self.byte_to_token_dict[bytes([x])] for x in text_as_bytes]

    def decode(self, encoded_bytes: list[TokenInt]) -> str:

        # given ids (list of integers), return Python string
        vocab_byte_string = b"".join(
            self.token_to_byte_dict[encoded_byte] for encoded_byte in encoded_bytes
        )

        # replace with special marker (�) for any bytes that can't be decoded
        # UTF-8 requires special start tokens for multi-byte
        # standard practice is to use `errors="replace"`
        text = vocab_byte_string.decode("utf-8", errors="replace")

        return text

    # for compatibility with tiktoken
    def decode_single_token_bytes(self, encoded_byte: TokenInt) -> bytes:

        return self.token_to_byte_dict[encoded_byte]
