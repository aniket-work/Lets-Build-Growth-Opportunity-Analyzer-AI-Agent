import re

def extract_csv_data(text):
    csv_pattern = re.compile(r"Date\s*,\s*Trend Score\s*\n(?:.*\n)+", re.IGNORECASE)
    match = csv_pattern.search(text)
    if match:
        return match.group()
    return None
