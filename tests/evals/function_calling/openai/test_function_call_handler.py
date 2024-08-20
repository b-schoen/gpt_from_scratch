import dataclasses

from gpt_from_scratch.evals.function_calling.openai import function_call_handler


@dataclasses.dataclass
class Function:
    """Mock of openai class"""

    arguments: str
    name: str


@dataclasses.dataclass
class ChatCompletionMessageToolCall:
    """Mock of openai class"""

    id: str
    type: str
    function: Function


def foo_1(bar: int, buzz: str = "cat") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: cat).

    """

    return f"foo1-{bar}-{buzz}"


def foo_2(bar: int, buzz: str = "cat") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: cat).

    """

    return f"foo2-{bar}-{buzz}"


def test_function_call_handler_resolve() -> None:
    """Check that handler works, works with multiple functions, and calls the correct one."""

    func_call_handler = function_call_handler.FunctionCallHandler(
        functions=[foo_1, foo_2]
    )

    tool_call = ChatCompletionMessageToolCall(
        id="call_x79a2rvl7MFr4CHKMgKr3cKn",
        function=Function(arguments='{"bar":23}', name="foo_2"),
        type="function",
    )

    result = func_call_handler._resolve_function_call(tool_call=tool_call)  # type: ignore

    # make sure we called `foo2`
    assert result == "foo2-23-cat"
