import torch

from jaxtyping import Float32


class DataLoaderLite:
    def __init__(
        self,
        B: int,
        T: int,
        tokens: Float32[torch.Tensor, "num_samples"],
    ) -> None:
        self.B = B
        self.T = T
        self.tokens = tokens

        print(f"loaded {len(self.tokens)} tokens")
        print(
            f"1 epoch = {len(self.tokens) // (B * T)} "
            "batches (steps to make one pass through data)"
        )

        # state
        self.current_position = 0

    def next_batch(
        self,
    ) -> tuple[Float32[torch.Tensor, "b t"], Float32[torch.Tensor, "b t"]]:

        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, loop back around to 0
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
