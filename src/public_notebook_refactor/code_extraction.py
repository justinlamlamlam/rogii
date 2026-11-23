import json

def extract_code_from_ipynb(input_path: str, output_path: str) -> None:
    """
    Extracts only the code cells from a .ipynb file and saves them into a .py file.

    Args:
        input_path (str): The path to the input .ipynb file.
        output_path (str): The path where the output .py file will be saved.

    Returns:
        None
    """
    with open(input_path, mode="r", encoding="utf-8") as file:
        notebook = json.load(file)

    code_lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            # Jointly join the source lines and add a newline between cells
            source = "".join(cell.get("source", []))
            if source.strip():
                code_lines.append(source)

    # Combine all blocks with double newlines for readability
    final_code = "\n\n".join(code_lines)

    with open(output_path, mode="w", encoding="utf-8") as file:
        file.write(final_code)

extract_code_from_ipynb(input_path="public_notebook_refactor/7-091-public.ipynb", output_path="extracted_code.py")