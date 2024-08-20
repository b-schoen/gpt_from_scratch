"""Functions around converting python callables to model tool use types."""

import inspect
from typing import Callable, Type, TypedDict, NotRequired
import enum

import anthropic
import openai

import docstring_parser


# TODO(bschoen): Add support for lists
def python_type_to_json_schema[T](typ: Type[T]) -> dict[str, str]:
    """
    Convert a Python type to a (non-recursive) JSON schema type.

    We simply don't handle the recursive part because I don't think OpenAI's function calling
    handles user defined types.

    """
    if typ == str:
        return {"type": "string"}
    elif typ == int:
        return {"type": "integer"}
    elif typ == float:
        return {"type": "number"}
    elif typ == bool:
        return {"type": "boolean"}
    else:
        raise ValueError(f"Unsupported type: {typ}")


# TODO(bschoen): Add `Literal` support / translating enums.`
# TODO(bschoen): Cap description at max size / check error
# TODO(bschoen): What you eventually probably want is some kind of intermediate format.
def generate_json_schema_for_function[
    R
](func: Callable[..., R]) -> anthropic.types.ToolParam:
    """
    Generate the JSON schema for a python Callable.

    Example:

        def get_weather(location: str, unit: str = "celsius") -> str:
            '''Get the current weather in a given location

            Args:
                location (str): The city and state, e.g. San Francisco, CA
                unit (str, optional): "The unit of temperature, either 'celsius' or 'fahrenheit'"

            '''

        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"
                    }
                },
                "required": ["location"]
            }
        }

    See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#specifying-tools

    """

    # Get function signature
    sig = inspect.signature(func)
    type_hints = inspect.get_annotations(func)

    # Get function name and docstring
    name = func.__name__
    parsed_docstring: docstring_parser.Docstring = docstring_parser.parse_from_object(
        func
    )

    # First line of docstring
    description = parsed_docstring.description

    if not description:
        raise ValueError(
            f"Function `{name}` must have a docstring with a description in the first line to be converted to a tool schema"
        )

    parsed_param_by_name: dict[str, docstring_parser.DocstringParam] = {
        x.arg_name: x for x in parsed_docstring.params
    }

    # Prepare parameters
    properties: dict[str, dict[str, str]] = {}
    required = []

    for param_name, param in sig.parameters.items():
        # determine type
        param_python_type = type_hints[param_name]
        param_schema = python_type_to_json_schema(param_python_type)

        # parse description from docstring
        param_schema["description"] = (
            parsed_param_by_name[param_name].description or f"Parameter: {param_name}"
        )

        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

        properties[param_name] = param_schema

    # Construct the schema
    schema: anthropic.types.ToolParam = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    return schema


def generate_openai_json_schema_for_function[
    R
](func: Callable[..., R]) -> openai.types.shared.FunctionDefinition:
    """
    Generate the JSON schema for a python Callable.
    """

    # Get function signature
    sig = inspect.signature(func)
    type_hints = inspect.get_annotations(func)

    # Get function name and docstring
    name = func.__name__
    parsed_docstring: docstring_parser.Docstring = docstring_parser.parse_from_object(
        func
    )

    # First line of docstring
    description = parsed_docstring.description

    parsed_param_by_name: dict[str, docstring_parser.DocstringParam] = {
        x.arg_name: x for x in parsed_docstring.params
    }

    # Prepare parameters
    properties: dict[str, openai.types.shared.FunctionParameters] = {}
    required = []

    for param_name, param in sig.parameters.items():
        # determine type
        param_python_type = type_hints[param_name]
        param_schema = python_type_to_json_schema(param_python_type)

        # parse description from docstring
        param_schema["description"] = (
            parsed_param_by_name[param_name].description or f"Parameter: {param_name}"
        )

        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            param_schema["default"] = param.default

        properties[param_name] = param_schema

    # Construct the schema
    schema: openai.types.shared.FunctionDefinition = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": False,  # strict set as `False` in order to allow parallel function calls
    }

    return schema
