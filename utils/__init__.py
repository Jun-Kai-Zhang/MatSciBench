import re

from utils.api import generate_with_api


def extract_final_answer(response: str) -> str:
    """Extract the last LaTeX boxed answer from a model response."""
    if not response:
        return ""

    matches = list(re.finditer(r"(?:\\)?boxed\s*\{", response))
    if not matches:
        return ""

    start_index = matches[-1].end()
    brace_count = 1
    end_index = start_index

    while brace_count > 0 and end_index < len(response):
        if response[end_index] == "{":
            brace_count += 1
        elif response[end_index] == "}":
            brace_count -= 1
        end_index += 1

    if brace_count > 0:
        return ""
    return response[start_index : end_index - 1].strip()
