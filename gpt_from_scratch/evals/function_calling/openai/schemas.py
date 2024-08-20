"""
Functions around converting python callables to model tool use types.

# TODO(bschoen): Combine this and the anthropic one
"""

import inspect
import typing
from typing import Callable, Type, TypedDict, NotRequired, Literal


import openai
import docstring_parser


class JsonSchemaDictFunctionParam(TypedDict):

    # todo(bschoen): Can this be an object?
    type: (
        Literal["string"] | Literal["integer"] | Literal["number"] | Literal["boolean"]
    )

    description: NotRequired[str]
    default: NotRequired[str | bool | int | float]


def python_type_to_json_schema[T](typ: Type[T]) -> JsonSchemaDictFunctionParam:
    """
    Convert a Python type to a (non-recursive) JSON schema type.

    We simply don't handle the recursive part yet, openai's does already though.

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


def generate_json_schema_for_function[
    R
](func: Callable[..., R]) -> openai.types.shared_params.FunctionDefinition:
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
    assert description

    parsed_param_by_name: dict[str, docstring_parser.DocstringParam] = {
        x.arg_name: x for x in parsed_docstring.params
    }

    # Prepare parameters
    properties: dict[str, openai.types.shared_params.FunctionParameters] = {}
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

        properties[param_name] = param_schema  # type: ignore

    # Construct the schema
    schema: openai.types.shared_params.FunctionDefinition = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,  # strict set as `False` in order to allow parallel function calls
    }

    return schema
