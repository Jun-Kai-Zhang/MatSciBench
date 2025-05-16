import re

def extract_final_answer(response: str) -> str:
    """
    Extracts the final answer from the model response.
    It searches for a \boxed command allowing optional whitespace between
    \boxed and {, and then correctly handles nested braces.
    """
    # Find \boxed{ pattern
    match_index = response.rfind("oxed{")
    if match_index == -1:
        return ""
    start_index = match_index + len("oxed{")
    
    # Find the matching closing brace by tracking nested braces
    brace_count = 1  # We've already found one opening brace
    end_index = start_index
    
    while brace_count > 0 and end_index < len(response):
        if response[end_index] == '{':
            brace_count += 1
        elif response[end_index] == '}':
            brace_count -= 1
        
        end_index += 1
        
        # Break if we've found the matching closing brace
        if brace_count == 0:
            break
    
    if brace_count > 0:  # Unbalanced braces
        return ""
    
    # Extract everything between \boxed{ and the matching }
    # Subtract 1 from end_index because the last increment puts it after the closing brace
    return response[start_index:end_index-1].strip()


def extract_number(text):
    # Capture numbers with optional decimal and scientific notation.
    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
