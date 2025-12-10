import numpy as np
from typing import Optional, List
import IPython.display as ipd


DICE_THROWING_TABLE = np.array(
    [
        [96, 22, 141, 41, 105, 122, 11, 30, 70, 121, 26, 9, 112, 49, 109, 14],
        [32, 6, 128, 63, 146, 46, 134, 81, 117, 39, 126, 56, 174, 18, 116, 83],
        [69, 95, 158, 13, 153, 55, 110, 24, 66, 139, 15, 132, 73, 58, 145, 79],
        [40, 17, 113, 85, 161, 2, 159, 100, 90, 176, 7, 34, 67, 160, 52, 170],
        [148, 74, 163, 45, 80, 97, 36, 107, 25, 143, 64, 125, 76, 13, 1, 93],
        [104, 157, 27, 167, 154, 68, 118, 91, 138, 71, 150, 29, 101, 162, 23, 151],
        [152, 60, 171, 53, 99, 133, 21, 127, 16, 155, 57, 175, 43, 168, 89, 172],
        [119, 84, 114, 50, 140, 86, 169, 94, 120, 88, 49, 38, 137, 148, 18, 47],
        [98, 142, 42, 156, 75, 129, 62, 123, 105, 45, 77, 19, 138, 118, 149, 8],
        [3, 87, 165, 61, 135, 47, 147, 33, 102, 4, 31, 164, 144, 59, 173, 78],
        [54, 130, 10, 103, 28, 37, 106, 5, 35, 20, 108, 92, 12, 124, 44, 131],
    ]
)


def numpy_to_markdown(
    array: np.ndarray,
    column_names: Optional[List[str]] = None,
    row_names: Optional[List[str]] = None,
) -> str:
    # Get the shape of the array
    rows, cols = array.shape

    # Create the header row
    if column_names is None:
        column_names = [f"Col {i}" for i in range(cols)]

    assert len(column_names) == cols

    header = "| " + " | " + " | ".join(column_names) + " |"
    separator = "| " + " | ".join(["---"] * (cols + 1)) + " |"

    if row_names is None:
        row_names = [f"Row {i}" for i in range(rows)]

    # Create the table rows
    table_rows = []
    for row, rn in zip(array, row_names):
        table_rows.append("| " + rn + "| " + " | ".join(map(str, row)) + " |")

    # Combine all parts
    markdown_table = "\n".join([header, separator] + table_rows)
    return markdown_table


def pretty_print_dice_throwing_table() -> ipd.Markdown:
    return ipd.Markdown(
        numpy_to_markdown(
            DICE_THROWING_TABLE,
            column_names=[f"dice roll {i}" for i in range(1, 17)],
            row_names=[f"dice sum {i}" for i in range(2, 13)],
        )
    )
