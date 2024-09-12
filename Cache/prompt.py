import re
# Replaces multiple leading and trailing whitespaces with a single space.
def compact_surrounding_spaces(text: str) -> str:
    return re.sub(r'^\s+|\s+$', ' ', text)


def compact_spaces(text: str) -> str:
    return ' '.join(text.split())