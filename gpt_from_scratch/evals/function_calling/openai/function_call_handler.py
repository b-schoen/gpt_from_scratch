from typing import Callable, Any
import json
import traceback

import openai

from . import schemas


"""

OpenAI function calling tips: https://platform.openai.com/docs/guides/function-calling/tips-and-best-practices

* Use enums for function arguments when possible
* Keep the number of functions low for higher accuracy (20)
* Set up evals to act as an aid in prompt engineering your function definitions and system messages
* Fine-tuning may help improve accuracy for function calling
* Turn on Structured Outputs by setting strict: "true"

Caveats:

* Configuring parallel function calling: Allows calling in parallel
* Parallel function calling disables structured outputs

"""


# TODO(bschoen): Support enums
class FunctionCallHandler:
    """
    Handles actually resolving / executing function calls.

    This way caller only needs to deal with python functions.

    """

    def __init__(self, functions: list[Callable[..., Any]]) -> None:

        self._function_name_to_function = {x.__name__: x for x in functions}

        # create `tools` arg schema once since used multiple times by calls to `create`
        self._schema_for_tools_arg: list[openai.types.chat.ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": schemas.generate_json_schema_for_function(x),
            }
            for x in functions
        ]

    def _resolve_function_call(
        self,
        tool_call: openai.types.chat.ChatCompletionMessageToolCall,
    ) -> Any:
        """Resolve and execute the actual function call represented by the tool call."""

        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

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
        tool_call: openai.types.chat.ChatCompletionMessageToolCall,
    ) -> openai.types.chat.ChatCompletionToolMessageParam:
        """Resolve the function call and convert it to a tool message."""

        # note: on error, we return the error to the model, in case it can
        #       do something about it
        try:
            function_result = self._resolve_function_call(tool_call=tool_call)
            function_result = str(function_result)
        except Exception as e:
            # on exception, convert to string so the model can handle it
            function_result = json.dumps(
                {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

        # TODO(bschoen): Why does the example include labeled outputs?
        return {
            "role": "tool",
            # always convert to string since that's format API expects
            "content": function_result,
            "tool_call_id": tool_call.id,
        }

    def get_schema_for_tools_arg(
        self,
    ) -> list[openai.types.chat.ChatCompletionToolParam]:

        return self._schema_for_tools_arg
