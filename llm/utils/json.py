
import re

def fix_json_string(s):
    # Replace single quotes with double quotes
    s = re.sub(r"(?<!\\)'", '"', s)

    # Fix unquoted keys (if any, rare in LLM outputs)
    s = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', s)

    # Balancing curly braces and square brackets
    def balance_braces_and_brackets(s):
        stack = []
        result = []
        
        for char in s:
            if char in ('{', '['):
                stack.append(char)
                result.append(char)
            elif char in ('}', ']'):
                if stack:
                    last_open = stack.pop()
                    if (last_open == '{' and char == '}') or (last_open == '[' and char == ']'):
                        result.append(char)
                    else:
                        # Mismatch case: close with appropriate brace
                        result.append('}' if last_open == '{' else ']')
                else:
                    # Unexpected closing, ignore or replace with closing brace
                    result.append('}')
            else:
                result.append(char)
        
        # Close any unmatched opening braces/brackets
        while stack:
            last_open = stack.pop()
            result.append('}' if last_open == '{' else ']')

        return ''.join(result)

    s = balance_braces_and_brackets(s)

    # Step 1: Add double quotes around the system_response value (assuming no nested quotes inside)
    s = re.sub(
        r'("system_response"\s*:\s*)([^"]([^,}]*))',
        lambda m: m.group(1) + '"' + m.group(2).strip() + '"',
        s,
        count=1,
    )
    
    # Step 2: Add double quotes around unquoted string values in key-value pairs (basic approximation)
    def quote_unquoted(match):
        val = match.group(1).strip()
        # Skip numbers, booleans, null, arrays, objects
        if re.match(r'^(\[.*\]|-?\d+(\.\d+)?|true|false|null)$', val, re.IGNORECASE):
            return ': ' + val
        if val.startswith('"') and val.endswith('"'):
            return ': ' + val
        return ': "' + val + '"'

    s = re.sub(
        r':\s*([^,\]\}\n]+)(?=[,\]\}])',
        quote_unquoted,
        s
    )
    
    # Step 3: Add double quotes around array string elements if missing
    def quote_array_items(match):
        items = match.group(1).split(',')
        new_items = []
        for item in items:
            item = item.strip()
            if not (item.startswith('"') and item.endswith('"')):
                item = f'"{item}"'
            new_items.append(item)
        return '[' + ', '.join(new_items) + ']'

    s = re.sub(
        r'\[\s*([^\]]*?)\s*\]',
        quote_array_items,
        s
    )

    return s
