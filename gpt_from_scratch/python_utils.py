"""Functions for extending generic python functionality."""

from typing import TypeVar, Callable, Any, ParamSpec, cast, Concatenate
import functools
import dataclasses
import json
import inspect
import math

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

# non-recursive json-type (since mypy doesn't really support recursive types well)
# intended only internally for use by functions in this file, not a general json type annotation
type _JsonDict = dict[str, Any] | list[Any] | str | int | float | bool | None


# TODO(bschoen): The complexity here is another case for these being actual
#                class instances, especially when need state management.
def wraps_partial(
    func: Callable[Concatenate[T, P], R],
    arg: T,
) -> Callable[P, R]:
    """`functools.partial` but extended to pass __doc__, __name__, and __annotations__ like `wraps`.

    Example:

        def tokenize_text(tokenizer: Tokenizer, text: str) -> list[str]:
            "Tokenize the text"
            ...

        tokenizer = Tokenizer()
        tokenize_text_fn = wraps_partial(tokenize_text, tokenizer=tokenizer)

        assert tokenize_text_fn.__doc__ == "Tokenize the text"

    Note:

        We can only do this one arg at a time, even for Python 3.12

        From https://peps.python.org/pep-0612

        > those that add/remove/change a variable number of parameters (for example,
          functools.partial will remain untypable even after this PEP)

    """

    # note: importantly, don't copy over annotations naively (like `functools.wraps` would
    #       naively do), as our signature has actually changed, as we do want
    #       that change reflected in the annotations, as that's what schema generation
    #       will use
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(arg, *args, **kwargs)

    # Override signature, removing first arg
    sig = inspect.signature(func)
    sig = sig.replace(parameters=tuple(sig.parameters.values())[1:])

    wrapper.__signature__ = sig

    return wrapper


def _dataclass_to_json_dict(obj: T) -> _JsonDict:
    """Convert a (potentially nested) dataclass to a json-serializable dictionary."""

    if dataclasses.is_dataclass(obj):
        return {
            field.name: _dataclass_to_json_dict(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
    elif isinstance(obj, list):
        return [_dataclass_to_json_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _dataclass_to_json_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def convert_return_value_to_json_string_wrapper(
    func: Callable[P, R]
) -> Callable[P, str]:
    """Wrapper to convert return value of a function to JSON, handling potentially nested dataclasses.

    Useful for the function calling APIs, which expect json strings, but we don't want
    all of our functions to return json strings since that'd be incredibly painful.

    This way we can have them return arbitrary dataclasses, and just wrap them in
    FunctionCallHandler.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        result = func(*args, **kwargs)
        result_dict = _dataclass_to_json_dict(result)

        # special case `str`, how do we want to handle this?
        # a bit confusing to the user to not always get a json deserializable string back
        # this is another point for not getting overly generic here, since it makes
        # for cute examples but is a bunch the user has to understand in practice
        if isinstance(result, str):
            return result

        # convert to json
        # note: encoded as utf-8 to preserve the original tokens verbatim
        # TODO(bschoen): there has to be a better pattern than `encode.decode`
        return (
            json.dumps(result_dict, indent=2, ensure_ascii=False)
            .encode("utf-8")
            .decode("utf-8")
        )

    return wrapper


def closest_power_of_two(n: int) -> int:
    # Find the power of 2 less than or equal to n
    lower = 2 ** math.floor(math.log2(n))

    # Find the power of 2 greater than n
    upper = lower * 2

    # Return the closest one
    return lower if (n - lower) < (upper - n) else upper


def next_power_of_two(n: int) -> int:

    # Find the power of 2 greater than n
    return 2 ** math.ceil(math.log2(n))


# TODO(bschoen): Make this just return the index so can actually slice
def get_first_n_examples(input_text: str, n: int, delimiter: str) -> str:
    """Useful for taking slice of text delimited by a special token."""

    examples = input_text.split(delimiter)

    # Return all text if n is greater than available examples
    if n > len(examples) - 1:
        return input_text

    result = delimiter.join(examples[:n]) + delimiter
    return result.strip()
