# we'll make a new provider with tool use
from typing import Callable, Any, ParamSpec
import dataclasses

from gpt_from_scratch.evals.function_calling.openai.function_call_handler import (
    FunctionCallHandler,
)
from gpt_from_scratch.evals.provider import Provider

import openai
import evalugator
import evalugator.api
import evalugator.api.providers

P = ParamSpec("P")


# note: task-standard/workbench/example-agents/fncall-baseline/commands.py
#       uses a `return` tool, which seems like an interesting technique

# note: subset of https://github.com/LRudL/evalugator/blob/main/evalugator/api/providers/openai.py#L15
#       that we care about that also supports tool use (not intended to be comprehensive)
#
# note: we can't just use `gpt-4o` etc directly since conflicts with the `openai` provider
_MODEL_ID_TO_MODEL_NAME: dict[str, str] = {
    f"function-calling-{model_name}": model_name
    for model_name in ["gpt-4o", "gpt-4o-mini"]
}


class MaxIterationsReachedError(Exception):
    pass


# TODO(bschoen): Might use `partial` here, since otherwise this is super inefficient
# TODO(bschoen): This really seems like it can only support single turn
# TODO(bschoen): Should we have a custom tool request message?
class OpenAIFunctionCallingProvider(Provider):
    """
    Individual instance of a provider supporting the specified functions.

    Example:
        import evalugator.api.dispatcher

        # we'll create a provider customized to have our tools
        provider = OpenAIFunctionCallingProvider(functions=[get_location, get_weather])

        evalugator.api.dispatcher.PROVIDERS.append(provider)

    """

    # note: it's actually fine for this to be stateful, but this whole design is a bit weird
    #       when it comes to customization points for the provider itself when it comes
    #       to provider specific things (which I guess is to be expected)
    #
    #       the provider itself essentially needs to be customizable, and capable of holding
    #       some immutable state, but not mutable per execution?
    def __init__(
        self, functions: list[Callable[..., Any]], max_iterations_per_execution: int = 3
    ) -> None:

        self._function_call_handler = FunctionCallHandler(functions=functions)

        # number of times to hand control back to the model after providing the result of a tool
        self._max_iterations_per_execution = max_iterations_per_execution

        # initialize client
        self._client = openai.OpenAI()

    def _get_model_name(self, model_id: str) -> str:
        return _MODEL_ID_TO_MODEL_NAME[model_id]

    def provides_model(self, model_id: str) -> bool:
        return any(model_id.startswith(prefix) for prefix in _MODEL_ID_TO_MODEL_NAME)

    def encode(self, model_id: str, *args: P.args, **kwargs: P.kwargs) -> str:
        return evalugator.api.providers.openai.encode(model_id, *args, **kwargs)

    def decode(self, model_id: str, *args: P.args, **kwargs: P.kwargs) -> str:
        return evalugator.api.providers.openai.decode(model_id, *args, **kwargs)

    # note: I guess response needs to contain all messages for future request? The
    #       `message` class isn't flexible enough to hold actual history though,
    #       so it seems like this thing overall isn't designed for multi-turn
    def execute(
        self,
        model_id: str,
        request: evalugator.api.requests.Request,
    ) -> evalugator.api.requests.Response:

        print("Executing function calling provider...")

        # list of {'role': ..., 'content': ...}
        messages = [dataclasses.asdict(x) for x in request.prompt]

        for i in range(self._max_iterations_per_execution):

            iteration_count = i + 1

            print(f"Attempt {iteration_count} / {self._max_iterations_per_execution}")

            # TODO(bschoen): Schemas actually _can_ contain nested types, if they're pydantic
            #                or dataclasses can support that easily, but description gets annoying,
            #                note that's for the input though.
            # TODO(bschoen): Response format is extremely valuable, can allow specifying it too
            # Note: Can't use response_format, structured_output, and parallel function calling together
            print("Creating completion...")
            response: openai.ChatCompletion = self._client.chat.completions.create(
                model=self._get_model_name(model_id),
                messages=messages,
                tools=self._function_call_handler.get_schema_for_tools_arg(),
            )

            # add message to history (true for both content and tool)
            assert len(response.choices) == 1

            choice: openai.types.CompletionChoice = response.choices[0]
            message: openai.types.chat.ChatCompletionMessage = choice.message

            messages.append(message.dict())

            # show any model text responses
            if message.content:
                print(f"Model response: {message.content}")

            # resolve any tool calls
            if message.tool_calls:

                print(f"Processing {len(message.tool_calls)} tool calls")

                for tool_call in message.tool_calls:

                    print(f"Resolving tool call in response: {tool_call.id}")
                    # print(f'Calling {tool_call.function.name} with args: {tool_call.function.arguments}')
                    tool_call_result_message = self._function_call_handler.resolve(
                        tool_call=tool_call
                    )

                    # print(f"Tool call result: {tool_call_result_message['content']}")

                    # add tool call result
                    print(f"Providing results from tool calls back to model...")
                    messages.append(tool_call_result_message)

            # otherwise, no tool calls left to resolve and we can return control back to the user
            else:
                print("\n---\nNo more tool calls left to resolve, breaking")

                # model_solution = messages[-1]['content']

                # print in case isn't valid json
                # print(f'Model solution: `{model_solution}`')

                # TODO(bschoen): could put parsed response_format in `context`
                # parse into dedicated struct
                # model_problem_solving_status = ModelProblemSolvingStatus.parse_raw(model_solution)

                return evalugator.api.requests.Response(
                    model_id=model_id,
                    request=request,
                    raw_responses=messages,
                    context=None,
                )

        # if we reach this point we've exhausted max iterations
        raise MaxIterationsReachedError(
            f"Exhausted max attempts: {self._max_iterations_per_execution}"
        )
