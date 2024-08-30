from gpt_from_scratch import tensor_utils

import torch


def test_debug_tensor() -> None:
    """Test that we can handle multidimensional tensors."""

    # using arbitrary shapes we commonly see
    for shape in [(32,), (8, 128), (3, 16, 16), (4, 8, 8, 16)]:

        # test float types
        tensor_float = torch.randn(shape)
        tensor_utils.debug_tensor(tensor_float)

        # test int types
        tensor_int = torch.randint(0, 100, shape)
        tensor_utils.debug_tensor(tensor_int)
