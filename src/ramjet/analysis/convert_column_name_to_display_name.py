"""
Code for converting a column name to a display name.
"""

import re


def convert_column_name_to_display_name(column_name: str) -> str:
    """
    Add method to convert from column style names to human-readable display names. Column style names should be
    snake_case with a double underscore before the units (if units are relevant). The display name will have the
    first letter capitalized, spaces between words, and the units in parentheses at the end.
    Example: `transit_duration__days` -> `Transit duration (days)`

    :param column_name: The column whose name should be converted.
    :return: The display version of the name.
    """
    display_name = re.sub(r"__(.*)", r" (\g<1>)", column_name)  # Move units into parentheses.
    display_name = display_name.replace("_", " ")
    specific_replacements = {
        r"tic id": "TIC ID",
        r"\(btjd\)": "(BTJD)",
        r"\(bjd\)": "(BJD)",
        r"\(jd\)": "(JD)",
        r"pdcsap": "PDCSAP",
        r"(^|\s)sap($|\s)": r"\g<1>SAP\g<2>",  # SAP avoiding possible cases where `sap` is part of a real word.
    }
    for pattern, replacement in specific_replacements.items():
        display_name = re.sub(pattern, replacement, display_name)
    display_name = display_name[0].upper() + display_name[1:]  # Make first letter uppercase.
    return display_name
