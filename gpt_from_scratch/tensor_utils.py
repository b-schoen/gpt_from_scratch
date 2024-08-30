"""Utilities for working with pytorch tensors."""

import torch

from typing import Callable, TypeVar

T = TypeVar("T")


def _format_float_list(float_list, precision=3):
    """Format a list of floats with specified precision."""
    # 6 = space between float
    return [f"{x:+.{precision}f}".rjust(precision + 6) for x in float_list]


def _format_tensor_values(tensor_slice, precision=3):
    """Format tensor values based on dtype."""
    if tensor_slice.dtype in [torch.float32, torch.float64]:
        return _format_float_list(tensor_slice.tolist(), precision)
    else:
        # 4 = space between ints
        return [f"{x:>4}" for x in tensor_slice.tolist()]


def debug_tensor(
    tensor: torch.Tensor,
    name: str = "Tensor",
    num_samples: int = 5,
):
    """
    Display debug information for a PyTorch tensor.

    Note:
        This is claude generated so messy, but probably not worth spending a lot
        of time refactoring a print function.

    Args:
        tensor (torch.Tensor): The tensor to debug.
        name (str): A name to identify the tensor (default: "Tensor").
        num_samples (int): Number of sample values to show per dimension (default: 3).

    """
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")

    if tensor.numel() == 0:
        print("  (Empty tensor)")
        return

    print("  Sample values:")

    if tensor.dim() == 1:
        samples = _format_tensor_values(tensor[:num_samples])
        suffix = "   ..." if tensor.shape[0] > num_samples else ""
        print(f"    {' '.join(samples)}{suffix}")

    elif tensor.dim() == 2:
        for i in range(min(num_samples, tensor.shape[0])):
            row_samples = _format_tensor_values(tensor[i, :num_samples])
            suffix = "   ..." if tensor.shape[1] > num_samples else ""
            print(f"    Row {i}: {' '.join(row_samples)}{suffix}")
        if tensor.shape[0] > num_samples:
            print("    ...")

    elif tensor.dim() == 3:
        for i in range(min(num_samples, tensor.shape[0])):
            print(f"    Slice {i}:")
            for j in range(min(num_samples, tensor.shape[1])):
                slice_samples = _format_tensor_values(tensor[i, j, :num_samples])
                print(f"      Row {j}: {' '.join(slice_samples)}   ...")
            if tensor.shape[1] > num_samples:
                print("      ...")
        if tensor.shape[0] > num_samples:
            print("    ...")

    elif tensor.dim() == 4:
        for i in range(min(num_samples, tensor.shape[0])):
            print(f"    Hyperslice {i}:")
            for j in range(min(num_samples, tensor.shape[1])):
                print(f"      Slice {j}:")
                for k in range(min(num_samples, tensor.shape[2])):
                    hyperslice_samples = _format_tensor_values(
                        tensor[i, j, k, :num_samples]
                    )
                    print(f"        Row {k}: {' '.join(hyperslice_samples)}   ...")
                if tensor.shape[2] > num_samples:
                    print("        ...")
            if tensor.shape[1] > num_samples:
                print("      ...")
        if tensor.shape[0] > num_samples:
            print("    ...")
    else:
        print("    (Tensor has more than 4 dimensions, showing first element)")
        print(
            f"    {' '.join(_format_tensor_values(tensor[(0,) * (tensor.dim() - 1)]))}"
        )
        print("    ...")


def apply_by_index_each_in_2d_tensor(
    tensor: torch.Tensor,
    func: Callable[[int, int], T],
) -> list[list[T]]:
    """
    Useful utility function for things like generating text annotations for a
    heatmap.

    Example:

        >>> tensor = # some 2d tensor representing values, where indices are vocab

        >>> text_annotations = tensor_utils.apply_by_index_each_in_2d_tensor(
              tensor,
              lambda i, j: f"{decode_single(i)} -> {decode_single(j)}",
            )

    """

    output = []

    for i in range(tensor.shape[0]):

        i_outputs = []

        for j in range(tensor.shape[1]):

            i_outputs.append(func(i, j))

        output.append(i_outputs)

    return output
