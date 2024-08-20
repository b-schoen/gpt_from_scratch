import pprint

from gpt_from_scratch.evals.function_calling.openai import schemas


def foo(bar: int, buzz: str = "cat") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: cat).

    """

    return f"foo-{bar}-{buzz}"


def test_generate_json_schema_for_function() -> None:

    pprint.pprint(schemas.generate_json_schema_for_function(foo))
