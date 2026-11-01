from pathlib import Path


def load_text_file(file_path: str) -> str:
    """
    Load the content of a text / md file.

    Args:
        file_path (str): The path to the text / md file.

    Returns:
        str: The content of the text / md file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if path.suffix.lower() not in {'.md', '.txt'}:
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .md and .txt files are supported.")
    
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError("The course-notes file is empty.")

    return text
