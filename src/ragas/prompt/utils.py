import copy
import typing as t
from typing import List, Type, Tuple, Dict

import dspy
from dspy import Signature, make_signature, OutputField, InputField
from pydantic import BaseModel
from pydantic.fields import FieldInfo


def get_all_strings(obj: t.Any) -> list[str]:
    """
    Get all strings in the objects.
    """
    strings = []

    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, BaseModel):
        for field_value in obj.model_dump().values():
            strings.extend(get_all_strings(field_value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(get_all_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(get_all_strings(value))

    return strings


def update_strings(obj: t.Any, old_strings: list[str], new_strings: list[str]) -> t.Any:
    """
    Replace strings in the object with new strings.
    Example Usage:
    ```
    old_strings = ["old1", "old2", "old3"]
    new_strings = ["new1", "new2", "new3"]
    obj = {"a": "old1", "b": "old2", "c": ["old1", "old2", "old3"], "d": {"e": "old2"}}
    update_strings(obj, old_strings, new_strings)
    ```
    """
    if len(old_strings) != len(new_strings):
        raise ValueError("The number of old and new strings must be the same")

    def replace_string(s: str) -> str:
        for old, new in zip(old_strings, new_strings):
            if s == old:
                return new
        return s

    if isinstance(obj, str):
        return replace_string(obj)
    elif isinstance(obj, BaseModel):
        new_obj = copy.deepcopy(obj)
        for field in new_obj.model_fields:
            setattr(
                new_obj,
                field,
                update_strings(getattr(new_obj, field), old_strings, new_strings),
            )
        return new_obj
    elif isinstance(obj, list):
        return [update_strings(item, old_strings, new_strings) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(update_strings(item, old_strings, new_strings) for item in obj)
    elif isinstance(obj, dict):
        return {k: update_strings(v, old_strings, new_strings) for k, v in obj.items()}

    return copy.deepcopy(obj)


def extract_json(text: str) -> str:
    """Identify json from a text blob by matching '[]' or '{}'.

    Warning: This will identify the first json structure!"""

    # check for markdown indicator; if present, start there
    md_json_idx = text.find("```json")
    if md_json_idx != -1:
        text = text[md_json_idx:]

    # search for json delimiter pairs
    left_bracket_idx = text.find("[")
    left_brace_idx = text.find("{")

    indices = [idx for idx in (left_bracket_idx, left_brace_idx) if idx != -1]
    start_idx = min(indices) if indices else None

    # If no delimiter found, return the original text
    if start_idx is None:
        return text

    # Identify the exterior delimiters defining JSON
    open_char = text[start_idx]
    close_char = "]" if open_char == "[" else "}"

    # Initialize a count to keep track of delimiter pairs
    count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            count += 1
        elif char == close_char:
            count -= 1

        # When count returns to zero, we've found a complete structure
        if count == 0:
            return text[start_idx: i + 1]

    return text  # In case of unbalanced JSON, return the original text



def create_dspy_example(user_input, response, refusal):
    return dspy.Example(user_input=user_input, response=response, refusal=refusal).with_inputs("user_input", "response")

def boolcheck(example, pred, trace=None):
    return int(example.refusal == pred.refusal)

def dspy_example_to_ragas(
        data: List,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel]
) -> List[Tuple[BaseModel, BaseModel]]:
    """
    Convert a list of data items to a list of tuples containing input and output models.

    Returns:
        List[Tuple[BaseModel, BaseModel]]: A list of tuples where each tuple contains an input model and an output model.
    """
    examples = []
    for item in data:
        example_in = input_model.model_validate(item.toDict())
        example_out = output_model.model_validate(item.toDict())
        examples.append((example_in, example_out))
    return examples


def base_model_to_signature(
        input: Type[BaseModel],
        output: Type[BaseModel],
        signature_name: str = "StringSignature",
        instructions: str = None,
) -> Type[Signature]:
    """
    Convert input and output BaseModel types of LLM metrics to a DSPy Signature.
    """
    signature_fields: Dict[str, Tuple[Type, FieldInfo]] = {}

    for name, field in input.model_fields.items():
        signature_fields[name] = (field.annotation, InputField(desc=field.description or ""))

    for name, field in output.model_fields.items():
        signature_fields[name] = (field.annotation, OutputField(desc=field.description or ""))

    return make_signature(signature_fields, instructions=instructions, signature_name=signature_name)


def ragas_example_to_dspy(
        data: List[Tuple[BaseModel, BaseModel]],
        input_model: Type[BaseModel],
        output_model: Type[BaseModel]
) -> List:
    '''Prepare training data for DSPy from existing ragas example '''
    examples = []
    for input_instance, output_instance in data:
        input_fields = input_model.model_validate(input_instance.model_dump()).model_dump()
        output_fields = output_model.model_validate(output_instance.model_dump()).model_dump()

        example = dspy.Example(
            **input_fields,
            **output_fields
        ).with_inputs(*input_fields.keys())

        examples.append(example)

    return examples


def judge(example, pred, trace=None):
    classifier = ", ".join(key for key, value in example.labels().toDict().items() if isinstance(value, int))
    example_verdict = example.labels().toDict().get(classifier)
    pred_verdict = pred.get(classifier)
    return int(example_verdict == pred_verdict)
