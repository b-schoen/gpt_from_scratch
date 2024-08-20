from typing import Callable, Any
import json
import traceback

import anthropic

from . import schemas


def _exception_to_json_dict[E: Exception](exception: E) -> dict[str, str]:
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc(),
    }


# TODO(bschoen): Support enums
# TODO(bschoen): If adding back openai support, make an ABC w/ type params for the tool use / tool result types
class FunctionCallHandler:
    """
    Handles actually resolving / executing function calls.

    This way caller only needs to deal with python functions.

    """

    def __init__(self, functions: list[Callable[..., Any]]) -> None:

        self._function_name_to_function = {x.__name__: x for x in functions}

        # create `tools` arg schema once since used multiple times by calls to `create`
        self._schema_for_tools_arg: list[anthropic.types.ToolParam] = [
            schemas.generate_json_schema_for_function(x) for x in functions
        ]

    def _resolve_function_call(
        self,
        tool_call: anthropic.types.ToolUseBlock,
    ) -> Any:
        """Resolve and execute the actual function call represented by the tool call."""

        function_name = tool_call.name
        function_args = tool_call.input

        if function_name not in self._function_name_to_function:
            raise KeyError(
                f"{function_name} not found in {self._function_name_to_function.keys()}"
            )

        func = self._function_name_to_function[function_name]

        # actually call the function
        result = func(**function_args)

        return result

    def resolve(
        self,
        tool_call: anthropic.types.ToolUseBlock,
    ) -> anthropic.types.ToolResultBlockParam:
        """Resolve the function call and convert it to a tool result message."""

        try:
            function_result = self._resolve_function_call(tool_call=tool_call)
            function_result = str(function_result)
            is_error = False
        except Exception as e:
            # on exception, convert to string so the model can handle it
            function_result = _exception_to_json_dict(e)
            function_result = json.dumps(function_result)
            is_error = True

        # TODO(bschoen): Why does the example include labeled outputs?
        # TODO(bschoen): Handle iterable content blocks
        output: anthropic.types.ToolResultBlockParam = {
            "type": "tool_result",
            # converting to string since that's expected format in API
            "content": function_result,
            "tool_use_id": tool_call.id,
            "is_error": is_error,
        }

        return output

    def get_schema_for_tools_arg(self) -> list[anthropic.types.ToolParam]:
        """Get the argument value needed for the `tools` parameter of the model's client completion call."""

        return self._schema_for_tools_arg
