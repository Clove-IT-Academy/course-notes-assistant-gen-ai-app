def clean_text(text: str) -> str:
    """Normalize line endings and remove outer blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip("\n")
