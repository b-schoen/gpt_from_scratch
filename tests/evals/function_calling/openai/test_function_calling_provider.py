from gpt_from_scratch.python_utils import (
    wraps_partial,
    convert_return_value_to_json_string_wrapper,
)

from gpt_from_scratch.tokenizer_utils import (
    get_detailed_and_complete_tokenization_info_for_text,
)

from gpt_from_scratch.evals.function_calling.openai.function_calling_provider import (
    OpenAIFunctionCallingProvider,
)

import evalugator
import tiktoken


def test_function_calling_provider_with_tokenization() -> None:

    # setup function

    # note: this is `gpt-4o`
    gpt4_tokenizer = tiktoken.get_encoding("o200k_base")

    # add tokenizer, so model doesn't have to know about it as an argument
    wrapped_with_tokenizer_fn = wraps_partial(
        get_detailed_and_complete_tokenization_info_for_text,
        gpt4_tokenizer,
    )

    # convert return type to JSON string since that's what model expects
    # TODO(bschoen): do this in `FunctionCallHandler.__init__` so it can just take functions
    #                without having to worry about this. That still seems like an additional
    #                thing the user has to learn
    wrapped_with_return_type_serialization_fn = (
        convert_return_value_to_json_string_wrapper(wrapped_with_tokenizer_fn)
    )

    openai_function_provider = OpenAIFunctionCallingProvider(
        functions=[wrapped_with_return_type_serialization_fn]
    )

    request = evalugator.api.requests.Request(
        prompt=[
            evalugator.api.requests.Message(
                role="user", content="How many 'r's are in: 你好 strawberry"
            ),
        ],
        context=None,
    )

    response = openai_function_provider.execute(
        model_id="function-calling-gpt-4o",
        request=request,
    )

    # TODO(bschoen): Actually evaluate the response
    print(response)
