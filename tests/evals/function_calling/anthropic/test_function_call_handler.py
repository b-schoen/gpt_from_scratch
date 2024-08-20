import json

import anthropic


from gpt_from_scratch.evals.function_calling.anthropic import function_call_handler


def foo_1(bar: int, buzz: str = "cat") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: cat).

    """

    return f"foo1-{bar}-{buzz}"


def foo_2(bar: int, buzz: str = "dog") -> str:
    """
    If the user provides a `bar` value, call this function with it and give them back the result.

    Args:
        bar (int): User provided value.
        buzz (str): Unused (default: dog).

    """

    return f"foo2-{bar}-{buzz}"


def divide_by_two_but_throw_exception_if_input_is_odd(value: int) -> str:
    """
    Divide the input value by two, but throw an exception if the input value is odd.

    Args:
        value (int): The value to divide by two.

    """

    if value % 2 == 1:
        raise ValueError("Value must be even")

    return value // 2


def test_function_call_handler_resolve() -> None:
    """Check that handler works, works with multiple functions, and calls the correct one."""

    func_call_handler = function_call_handler.FunctionCallHandler(
        functions=[foo_1, foo_2]
    )

    tool_call = anthropic.types.ToolUseBlock(
        type="tool_use",
        id="fake_tool_use_id",
        name="foo_2",
        input={"bar": 23},
    )

    result = func_call_handler._resolve_function_call(tool_call=tool_call)  # type: ignore

    # make sure we called `foo2`
    assert result == "foo2-23-dog"


def test_function_call_handler_resolve_with_and_without_error() -> None:
    """Check that handler works when error."""

    func_call_handler = function_call_handler.FunctionCallHandler(
        functions=[foo_1, divide_by_two_but_throw_exception_if_input_is_odd]
    )

    tool_call = anthropic.types.ToolUseBlock(
        type="tool_use",
        id="fake_tool_use_id",
        name="divide_by_two_but_throw_exception_if_input_is_odd",
        input={"value": 10},
    )

    result = func_call_handler.resolve(tool_call=tool_call)

    assert result["content"] == "5"
    assert result["is_error"] == False

    # now try with odd input, which should result in an error
    tool_call.input = {"value": 11}

    result = func_call_handler.resolve(tool_call=tool_call)

    # check exception type, message, and part of stacktrace (function name) are in output
    assert result["is_error"] == True
    for expected_string in [
        "ValueError",
        "Value must be even",
        "divide_by_two_but_throw_exception_if_input_is_odd",
    ]:
        assert expected_string in result["content"]

    # check the exception is loadable as a json dict
    exception_json_dict = json.loads(result["content"])

    assert exception_json_dict["type"] == "ValueError"
    assert exception_json_dict["message"] == "Value must be even"
    assert exception_json_dict["traceback"] is not None
