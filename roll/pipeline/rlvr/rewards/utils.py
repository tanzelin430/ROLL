"""
Utility functions for reward workers.

Shared extraction and processing functions used by multiple reward workers.
"""
import re
from typing import Optional, List


def extract_last_boxed(string: str) -> Optional[str]:
    """
    Extract the content of the last \\boxed{} or \\fbox{} in a string.

    Uses brace-matching algorithm to correctly handle nested braces,
    e.g., \\boxed{\\frac{1}{2}} -> "\\frac{1}{2}"

    This function is more robust than regex-based approaches because:
    1. It doesn't depend on separators like \\n\\n or </think>
    2. It correctly handles nested braces in math expressions
    3. It works with both display math \\[\\boxed{...}\\] and inline \\boxed{...}

    Args:
        string: The input string to search

    Returns:
        The content inside the last \\boxed{} or \\fbox{}, or None if not found.
        The content is stripped of leading/trailing whitespace.

    Examples:
        >>> extract_last_boxed("The answer is \\\\boxed{42}")
        '42'
        >>> extract_last_boxed("\\\\boxed{\\\\frac{1}{2}}")
        '\\\\frac{1}{2}'
        >>> extract_last_boxed("First \\\\boxed{0}, then \\\\boxed{1}")
        '1'
        >>> extract_last_boxed("No boxed content")
        None
    """
    # Find the last occurrence of \boxed or \fbox
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    # Find matching braces using a counter
    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1:right_brace_idx].strip()


def extract_answer_tags(text: str) -> Optional[str]:
    """
    Extract content from the last <answer>...</answer> tags.

    Used for Zebra Puzzle and similar structured output formats.

    Args:
        text: The input string to search

    Returns:
        The content inside the last <answer>...</answer> tags,
        or None if not found.

    Examples:
        >>> extract_answer_tags("Some text <answer>{'key': 'value'}</answer>")
        "{'key': 'value'}"
        >>> extract_answer_tags("No answer tags here")
        None
    """
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_code_blocks(text: str, language: str = None) -> str:
    """
    Extract code from markdown code blocks.

    Handles ```python ... ``` or ``` ... ``` blocks.
    If multiple blocks exist, they are joined with newlines.
    Also handles </think> tag removal for chain-of-thought models.

    Args:
        text: The input string containing markdown code blocks
        language: Optional language filter (e.g., "python"). If None, matches any language.

    Returns:
        Extracted code as a string, or empty string if no blocks found.

    Examples:
        >>> extract_code_blocks("```python\\ndef foo(): pass\\n```")
        'def foo(): pass'
        >>> extract_code_blocks("No code here")
        ''
    """
    # Remove content before </think> if present (for chain-of-thought models)
    if '</think>' in text:
        text = text.split('</think>')[-1]

    # Build regex pattern based on language filter
    if language:
        pattern = rf'```{language}\s*\n(.*?)\n```'
    else:
        pattern = r'```(?:\w+)?\s*\n(.*?)\n```'

    matches = re.findall(pattern, text, re.DOTALL)
    return '\n'.join(matches).strip() if matches else ''


def extract_option_letter(text: str, valid_options: str = 'ABCDEFGHIJ') -> Optional[str]:
    """
    Extract a single option letter from text.

    Tries multiple extraction strategies:
    1. From \\boxed{} content
    2. From "Final Answer:" pattern
    3. From "The answer is:" pattern
    4. First valid option letter found in text

    Args:
        text: The input string to search
        valid_options: String of valid option letters (default: A-J)

    Returns:
        Single uppercase option letter, or None if not found.

    Examples:
        >>> extract_option_letter(r"The answer is \\boxed{B}")
        'B'
        >>> extract_option_letter("Final Answer: C")
        'C'
    """
    # Strategy 1: Try \boxed{} first
    boxed = extract_last_boxed(text)
    if boxed:
        for char in boxed.upper():
            if char in valid_options:
                return char

    # Strategy 2: "Final Answer:" pattern
    match = re.search(r'Final\s+Answer[:\s]+([A-J])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 3: "The answer is:" pattern
    match = re.search(r'[Tt]he\s+answer\s+is[:\s]+([A-J])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 4: Find any valid option in text (last occurrence)
    for char in reversed(text.upper()):
        if char in valid_options:
            return char

    return None
